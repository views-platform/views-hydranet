"VolumeHandler: Authoritative Layout Management for Spatiotemporal Volumes."

import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class SpatialConvention(Enum):
    GEOGRAPHIC = "geographic"
    NORTH_UP = "north_up"


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
    spatial_cols: Tuple[str, ...]  # Usually (row_col, col_col)

    # Classification
    identity_cols: Tuple[str, ...]
    feature_cols: Tuple[str, ...]

    spatial_offset: Tuple[int, int]
    history: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    spatial_convention: SpatialConvention = SpatialConvention.GEOGRAPHIC


class VolumeHandler:
    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor],
        axes: Union[List[str], Tuple[str, ...]],
        channel_map: Union[List[str], Tuple[str, ...]],
        time_col: str,
        id_col: str,
        spatial_cols: Union[List[str], Tuple[str, ...]],
        identity_cols: Union[List[str], Tuple[str, ...]] = (),
        feature_cols: Union[List[str], Tuple[str, ...]] = (),
        spatial_offset: Tuple[int, int] = (0, 0),
        config: Optional[Dict[str, Any]] = None,
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
            spatial_offset=spatial_offset,
        )

        # 1. Validation: Channel dimension must match channel_map
        c_idx = self.get_axis_idx("C")
        actual_channels = self._data.shape[c_idx]
        expected_channels = len(self.channel_map)
        if actual_channels != expected_channels:
            err_msg = (
                f"VolumeHandler: Channel mismatch! Data has {actual_channels} channels, "
                f"but channel_map has {expected_channels} names."
            )

            logger.error(err_msg)

            raise ValueError(err_msg)

        # 2. ADR 046: Execute derivations (if instructions provided)
        if config and "derivations" in config:
            self._execute_derivations(config["derivations"])

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> "VolumeHandler":
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
            err_msg = (
                f"VolumeHandler Contract Violation: Missing Ledger Role {e} in config.\n"
                f"To comply with ADR 007, your config must define:\n"
                f"  'time_col': The temporal index (e.g., 'month_id')\n"
                f"  'id_col':   The unit index (e.g., 'priogrid_gid')\n"
                f"  'spatial_cols': ['row_col', 'col_col']\n"
            )

            logger.error(err_msg)

            raise KeyError(err_msg)

        identity_cols = config.get("identity_cols", [])
        feature_cols = config.get("features", [])

        # --- THE STRICT HANDSHAKE (ADR 007 Section 1.2) ---
        required_roles = [time_col, id_col, y_col, x_col]
        # Offsets are also strictly required now
        try:
            row_offset = config["row_offset"]
            col_offset = config["col_offset"]
        except KeyError as e:
            err_msg = f"VolumeHandler Contract Violation: Missing mandatory offset {e} in config."
            logger.error(err_msg)
            raise KeyError(err_msg)

        all_required = list(set(required_roles + list(identity_cols) + list(feature_cols)))

        missing = [c for c in all_required if c not in df.columns]
        if missing:
            err_msg = f"VolumeHandler Handshake Failed! Missing columns: {missing}"

            logger.error(err_msg)

            raise ValueError(err_msg)

        # The channel map is ordered: [Primary Keys] + [Metadata] + [Features]
        # We ensure time_col and id_col are always first.
        channel_map = [time_col, id_col]
        for col in list(identity_cols) + list(feature_cols):
            if col not in channel_map:
                channel_map.append(col)

        # 2. Structural Anchoring
        month_min = df[time_col].min()
        month_max = df[time_col].max()
        month_range = int(month_max - month_min + 1)

        # 3. Coordinate Calculation
        r_idx = (df[y_col] - row_offset).astype(int).values
        c_idx = (df[x_col] - col_offset).astype(int).values
        m_idx = (df[time_col] - month_min).astype(int).values

        # --- SPATIAL INTEGRITY GUARD (ADR 012 / Test Remediation Plan 2026-02-19) ---
        # 1. Lower Bound: Prevent silent numpy wrap-around (scrambling)
        if r_idx.min() < 0:
            err_msg = (
                f"VolumeHandler Spatial Anchor Violation: row indices are negative after "
                f"applying row_offset={row_offset}. "
                f"df['{y_col}'].min()={df[y_col].min()} → "
                f"min r_idx={r_idx.min()}. "
                f"row_offset must be <= df['{y_col}'].min(). "
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        if c_idx.min() < 0:
            err_msg = (
                f"VolumeHandler Spatial Anchor Violation: col indices are negative after "
                f"applying col_offset={col_offset}. "
                f"df['{x_col}'].min()={df[x_col].min()} → "
                f"min c_idx={c_idx.min()}. "
                f"col_offset must be <= df['{x_col}'].min(). "
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        # 2. Upper Bound: Prevent out-of-bounds writing/crashes
        if r_idx.max() >= height:
            err_msg = (
                f"VolumeHandler Spatial Span Violation: row index {r_idx.max()} exceeds "
                f"volume height {height}. "
                f"df['{y_col}'].max()={df[y_col].max()} | row_offset={row_offset}. "
                f"Increase config['height'] or adjust offsets."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        if c_idx.max() >= width:
            err_msg = (
                f"VolumeHandler Spatial Span Violation: col index {c_idx.max()} exceeds "
                f"volume width {width}. "
                f"df['{x_col}'].max()={df[x_col].max()} | col_offset={col_offset}. "
                f"Increase config['width'] or adjust offsets."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        # 4. Allocation & Population
        vol = np.zeros([height, width, month_range, len(channel_map)], dtype=np.float32)

        # Dense Identity Population (Temporal)
        m_chan_idx = channel_map.index(time_col)
        m_vals_global = np.arange(month_min, month_max + 1)
        vol[..., m_chan_idx] = m_vals_global.reshape(1, 1, month_range)

        for i, col_name in enumerate(channel_map):
            vol[r_idx, c_idx, m_idx, i] = df[col_name].values

        # 5. Flip & Layout
        vol = np.flip(vol, axis=0)  # North-Up
        vol = np.transpose(vol, (2, 0, 1, 3))  # [T, H, W, C]

        mem_mb = vol.nbytes / (1024**2)
        logger.debug(
            f"🌐 VolumeHandler: Created Global Volume {vol.shape} | Memory: {mem_mb:.2f} MB"
        )

        result = cls(
            data=vol,
            axes=("T", "H", "W", "C"),
            channel_map=channel_map,
            time_col=time_col,
            id_col=id_col,
            spatial_cols=(y_col, x_col),
            identity_cols=identity_cols,
            feature_cols=feature_cols,
            spatial_offset=(row_offset, col_offset),
            config=config,
        )
        result._metadata = replace(result._metadata, spatial_convention=SpatialConvention.NORTH_UP)
        return result

    def to_pytorch(self, device: torch.device, include_identities: bool = False) -> torch.Tensor:
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
            feature_indices = [
                i for i, name in enumerate(self.channel_map) if name in self.feature_cols
            ]
            assert feature_indices, (
                "VolumeHandler.to_pytorch: feature_cols is empty — "
                "handler was not constructed via from_df or config['features'] is missing."
            )
            np_data = np_data[:, :, :, feature_indices]

        tensor = torch.from_numpy(np_data).to(device)
        tensor = tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
        tensor = tensor.unsqueeze(0)  # [B, T, C, H, W]

        return tensor

    def wrap_predictions(
        self, posterior_data: Union[np.ndarray, torch.Tensor], target_names: List[str]
    ) -> "VolumeHandler":
        """
        Creates a new VolumeHandler for model outputs, anchored to this handler's ledger.
        Automatically applies ADR 032 naming Engine and Watermarks the volume with IDs.
        """
        # 1. Automated Naming (Internal Symmetry Gate)
        # ADR 032: Naming is derived literally from target names.
        full_signal_names = [f"{PRED_PREFIX}{n}" for n in target_names]

        # 2. THE WATERMARK (Red Team Hardening)
        # We prepend ALL identity channels from the parent to the prediction data
        # so the prediction volumes are "Self-Describing" and Join-Safe.
        time_col = self._metadata.time_col
        id_col = self._metadata.id_col

        # Identify primary join keys from the parent
        try:
            head_time_idx = self.channel_map.index(time_col)
            head_id_idx = self.channel_map.index(id_col)
        except ValueError as e:
            err_msg = (
                f"VolumeHandler Ledger Corruption: Primary index '{e}' missing from channel map. "
                f"Prediction wrapping requires watermarked primary keys."
            )

            logger.error(err_msg)

            raise ValueError(err_msg)

        # Identify all non-feature channels from parent (excluding primary keys already handled)
        identity_names = [
            n for n in self.channel_map if n in self.identity_cols and n not in [time_col, id_col]
        ]
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
            err_msg = (
                f"VolumeHandler Contract Violation: Signal duration ({work_data.shape[0]}) "
                f"does not match Handler duration ({self.shape[0]})."
            )

            logger.error(err_msg)

            raise ValueError(err_msg)

        # 4. Concatenate Identity Watermarks
        # Primary keys (Time, ID) must come first to ensure Join-Safety
        primary_idxs = [head_time_idx, head_id_idx]
        primary_names = [time_col, id_col]

        axes: Tuple[str, ...]
        if work_data.ndim == 5:
            # Stochastic: [T, H, W, C, S]
            axes = ("T", "H", "W", "C", "S")
            n_samples = work_data.shape[-1]

            # Extract and repeat identity watermarks.
            # Ensure float32 consistency: self.data is float32 (from_df),
            # work_data is float32 (PyTorch output). Defensive cast to prevent
            # accidental upcast if upstream ever introduces float64.
            # Identity values (priogrid_gid, month_id, row, col) are small integers
            # exactly representable in float32 (exact up to 2^24 ≈ 16.7 million).
            id_vols = []
            for idx in primary_idxs + identity_idxs:
                slice_data = self.data[..., idx : idx + 1]  # [T, H, W, 1]
                watermark = np.expand_dims(slice_data, axis=-1)  # [T, H, W, 1, 1]
                watermark = np.repeat(watermark, n_samples, axis=-1)  # [T, H, W, 1, S]
                id_vols.append(watermark.astype(np.float32))

            # work_data from PyTorch is float32; ensure consistent output dtype.
            full_data = np.concatenate(id_vols + [work_data.astype(np.float32)], axis=-2)
        else:
            # Point: [T, H, W, C]
            axes = ("T", "H", "W", "C")
            id_vols = [self.data[..., idx : idx + 1] for idx in primary_idxs + identity_idxs]
            full_data = np.concatenate(id_vols + [work_data], axis=-1)

        # The final channel map is [Keys] + [Identities] + [Predictions]
        final_names = primary_names + identity_names + full_signal_names

        result = VolumeHandler(
            data=full_data,
            axes=axes,
            channel_map=final_names,
            time_col=time_col,
            id_col=id_col,
            spatial_cols=self.spatial_cols,
            identity_cols=tuple(identity_names),
            feature_cols=tuple(full_signal_names),
            spatial_offset=self.spatial_offset,
        )
        result._metadata = replace(
            result._metadata, spatial_convention=self._metadata.spatial_convention
        )
        return result

    def collapse_to_point(self, method: str) -> "VolumeHandler":
        """
        Mathematically collapses the sample dimension ('S') into a point estimate.
        Governed by ADR 021: Volume Dimension Reduction.
        """
        if "S" not in self._metadata.axes:
            logger.warning(
                "VolumeHandler: collapse_to_point() called on already 4D volume. Skipping."
            )
            return self

        s_idx = self.get_axis_idx("S")
        logger.info(
            f"📍 VolumeHandler: Collapsing dimension 'S' via {method} (ADR 021 Survival Gate)"
        )

        if torch.is_tensor(self._data):
            work_data = self._data.detach().cpu().numpy()
        else:
            work_data = self._data

        if method in ["arithmetic_mean", "mean"]:
            collapsed_data = np.mean(work_data, axis=s_idx)
        elif method == "median":
            collapsed_data = np.median(work_data, axis=s_idx)
        else:
            err_msg = (
                f"Collapse method '{method}' is not defined in ADR 021. "
                "Must be 'arithmetic_mean' or 'median'."
            )

            logger.error(err_msg)

            raise NotImplementedError(err_msg)

        # Update axes: Filter out 'S'
        new_axes = tuple(ax for ax in self._metadata.axes if ax != "S")

        result = VolumeHandler(
            data=collapsed_data,
            axes=new_axes,
            channel_map=self.channel_map,
            time_col=self.time_col,
            id_col=self.id_col,
            spatial_cols=self.spatial_cols,
            identity_cols=self.identity_cols,
            feature_cols=self.feature_cols,
            spatial_offset=self.spatial_offset,
        )
        result._metadata = replace(
            result._metadata, spatial_convention=self._metadata.spatial_convention
        )
        return result

    def slice_time(self, start_idx: int, end_idx: int) -> "VolumeHandler":
        """
        Returns a new VolumeHandler containing a temporal subset of the data.
        """
        t_idx = self.get_axis_idx("T")
        max_t = self._data.shape[t_idx]

        if start_idx < 0 or end_idx > max_t or start_idx >= end_idx:
            err_msg = (
                f"VolumeHandler Contract Violation: Invalid time slice [{start_idx}:{end_idx}] "
                f"for volume with duration {max_t}."
            )

            logger.error(err_msg)

            raise ValueError(err_msg)

        slices = [slice(None)] * self._data.ndim
        slices[t_idx] = slice(start_idx, end_idx)
        new_data = self._data[tuple(slices)]

        result = VolumeHandler(
            data=new_data,
            axes=self.axes,
            channel_map=self.channel_map,
            time_col=self.time_col,
            id_col=self.id_col,
            spatial_cols=self.spatial_cols,
            identity_cols=self.identity_cols,
            feature_cols=self.feature_cols,
            spatial_offset=self.spatial_offset,
        )
        result._metadata = replace(
            result._metadata, spatial_convention=self._metadata.spatial_convention
        )
        return result

    def extrapolate_time(self, steps: int) -> "VolumeHandler":
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

        m_col = self._metadata.time_col
        m_idx = self.channel_map.index(m_col)
        if torch.is_tensor(self._data):
            t_increments = torch.arange(1, steps + 1, device=self._data.device).view(steps, 1, 1)
            t_future_vol = cast(torch.Tensor, future_vol)
            t_future_vol[..., m_idx] += t_increments
        else:
            np_increments = np.arange(1, steps + 1).reshape(steps, 1, 1)
            np_future_vol = cast(np.ndarray, future_vol)
            np_future_vol[..., m_idx] += np_increments

        result = VolumeHandler(
            data=future_vol,
            axes=self.axes,
            channel_map=self.channel_map,
            time_col=self.time_col,
            id_col=self.id_col,
            spatial_cols=self.spatial_cols,
            identity_cols=self.identity_cols,
            feature_cols=self.feature_cols,
            spatial_offset=self.spatial_offset,
        )
        result._metadata = replace(
            result._metadata, spatial_convention=self._metadata.spatial_convention
        )
        return result

    def _permute(self, dims: Union[List[int], Tuple[int, ...]]) -> "VolumeHandler":
        """
        Returns a new VolumeHandler with reordered axes.
        """
        dims_tuple = tuple(dims)
        if torch.is_tensor(self._data):
            new_data: Any = (
                cast(torch.Tensor, self._data).permute(*dims_tuple).contiguous().clone()
            )
        else:
            new_data = np.transpose(cast(np.ndarray, self._data), dims_tuple).copy()

        new_axes = tuple(self._metadata.axes[i] for i in dims_tuple)
        result = VolumeHandler(
            data=new_data,
            axes=new_axes,
            channel_map=self.channel_map,
            time_col=self.time_col,
            id_col=self.id_col,
            spatial_cols=self.spatial_cols,
            identity_cols=self.identity_cols,
            feature_cols=self.feature_cols,
            spatial_offset=self.spatial_offset,
        )
        result._metadata = replace(
            result._metadata,
            history=self._metadata.history + (("permute", dims_tuple),),
            spatial_convention=self._metadata.spatial_convention,
        )
        return result

    def flip(self, axis_label: str) -> "VolumeHandler":
        """
        Returns a new VolumeHandler flipped along a named axis.
        """
        idx = self.get_axis_idx(axis_label)
        if torch.is_tensor(self._data):
            new_data: Any = torch.flip(cast(torch.Tensor, self._data), dims=[idx])
        else:
            new_data = np.flip(cast(np.ndarray, self._data), axis=idx).copy()

        result = VolumeHandler(
            data=new_data,
            axes=self.axes,
            channel_map=self.channel_map,
            time_col=self.time_col,
            id_col=self.id_col,
            spatial_cols=self.spatial_cols,
            identity_cols=self.identity_cols,
            feature_cols=self.feature_cols,
            spatial_offset=self.spatial_offset,
        )
        result._metadata = replace(
            result._metadata,
            history=self._metadata.history + (("flip", axis_label),),
            spatial_convention=self._metadata.spatial_convention,
        )
        return result

    @property
    def data(self) -> Union[np.ndarray, "torch.Tensor"]:
        return self._data

    @property
    def shape(self) -> Tuple:
        return self._data.shape

    def __len__(self) -> int:
        return self._data.shape[self.get_axis_idx("T")]

    @property
    def axes(self) -> Tuple[str, ...]:
        return self._metadata.axes

    @property
    def channel_map(self) -> Tuple[str, ...]:
        return self._metadata.channel_map

    @property
    def id_col(self) -> str:
        return self._metadata.id_col

    @property
    def time_col(self) -> str:
        return self._metadata.time_col

    @property
    def spatial_cols(self) -> Tuple[str, ...]:
        return self._metadata.spatial_cols

    @property
    def spatial_offset(self) -> Tuple[int, int]:
        return self._metadata.spatial_offset

    @property
    def feature_cols(self) -> Tuple[str, ...]:
        return self._metadata.feature_cols

    @property
    def identity_cols(self) -> Tuple[str, ...]:
        return self._metadata.identity_cols

    @property
    def history(self) -> Tuple[Tuple[str, Any], ...]:
        return self._metadata.history

    @property
    def spatial_convention(self) -> SpatialConvention:
        return self._metadata.spatial_convention

    # Cross-ref: DataFetcher.apply_blueprint() (data_fetcher.py)
    # Both paths RAISE if source is missing — see tests/test_derivation_parity.py.
    def _execute_derivations(self, derivations_config: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Executes additive feature engineering instructions (ADR 046).
        Mutates internal _data and _metadata.
        """
        c_idx = self.get_axis_idx("C")
        new_channels = list(self._metadata.channel_map)
        new_features = list(self.feature_cols)

        for op, instructions in derivations_config.items():
            for instr in instructions:
                src_name = instr.get("from")
                dst_name = instr.get("to")

                # Zero Magic: Check mandatory keys
                if not all([src_name, dst_name]):
                    err_msg = (
                        f"VolumeHandler Derivation Error: Missing mandatory keys (from, to) "
                        f"in {op} instr: {instr}"
                    )
                    logger.error(err_msg)
                    raise KeyError(err_msg)

                if src_name not in self.channel_map:
                    err_msg = (
                        f"VolumeHandler Derivation Error: Source feature '{src_name}' "
                        "not found in volume."
                    )
                    logger.error(err_msg)
                    raise ValueError(err_msg)

                src_idx = self.channel_map.index(src_name)

                # Slicing the volume on the channel axis
                slc: List[Union[slice, int]] = [slice(None)] * self._data.ndim
                slc[c_idx] = src_idx

                derived_data: Union[np.ndarray, torch.Tensor]
                if op == "binary":
                    # Zero Magic: threshold is mandatory for binary
                    if "threshold" not in instr:
                        err_msg = (
                            f"VolumeHandler Derivation Error: 'binary' op requires explicit "
                            f"'threshold' key in {instr}"
                        )
                        logger.error(err_msg)
                        raise KeyError(err_msg)

                    threshold = instr["threshold"]

                    # Perform thresholding
                    if torch.is_tensor(self._data):
                        derived_data = (
                            (self._data[tuple(slc)] > threshold).float().unsqueeze(c_idx)
                        )
                    else:
                        # Use a temporary variable to help mypy with types
                        raw_slice = self._data[tuple(slc)]
                        assert isinstance(raw_slice, np.ndarray)
                        derived_data = (raw_slice > threshold).astype(np.float32)
                        derived_data = np.expand_dims(derived_data, axis=c_idx)
                else:
                    err_msg = (
                        f"VolumeHandler Derivation Error: Operation '{op}' is not implemented."
                    )
                    logger.error(err_msg)
                    raise NotImplementedError(err_msg)

                # Concatenate to data
                if torch.is_tensor(self._data):
                    assert torch.is_tensor(derived_data)
                    self._data = torch.cat([self._data, derived_data], dim=c_idx)
                else:
                    assert isinstance(derived_data, np.ndarray)
                    assert isinstance(self._data, np.ndarray)
                    self._data = np.concatenate([self._data, derived_data], axis=c_idx)

                # Update ledger
                new_channels.append(cast(str, dst_name))
                new_features.append(cast(str, dst_name))

        self._metadata = replace(
            self._metadata, channel_map=tuple(new_channels), feature_cols=tuple(new_features)
        )

    def get_axis_idx(self, label: str) -> int:
        return self._metadata.axes.index(label)
