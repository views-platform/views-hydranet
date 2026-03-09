"VolumeHandler: Authoritative Layout Management for Spatiotemporal Volumes."

import gc
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import polars as pl
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
    spatial_cols: Tuple[str, ...]  # Usually (row_col, col_col)

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
        vol = np.zeros([height, width, month_range, len(channel_map)], dtype=np.float64)

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

        return cls(
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
                i for i, name in enumerate(self.channel_map) if name in self._metadata.feature_cols
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
            raise ValueError(
                f"VolumeHandler Ledger Corruption: Primary index '{e}' missing from channel map. "
                f"Prediction wrapping requires watermarked primary keys."
            )

        # Identify all non-feature channels from parent (excluding primary keys already handled)
        identity_names = [
            n
            for n in self.channel_map
            if n in self._metadata.identity_cols and n not in [time_col, id_col]
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
            raise ValueError(
                f"VolumeHandler Contract Violation: Signal duration ({work_data.shape[0]}) "
                f"does not match Handler duration ({self.shape[0]})."
            )

        # 4. Concatenate Identity Watermarks
        # Primary keys (Time, ID) must come first to ensure Join-Safety
        primary_idxs = [head_time_idx, head_id_idx]
        primary_names = [time_col, id_col]

        axes: Tuple[str, ...]
        if work_data.ndim == 5:
            # Stochastic: [T, H, W, C, S]
            axes = ("T", "H", "W", "C", "S")
            n_samples = work_data.shape[-1]

            # Extract and repeat identity watermarks
            id_vols = []
            for idx in primary_idxs + identity_idxs:
                slice_data = self.data[..., idx : idx + 1]  # [T, H, W, 1]
                watermark = np.expand_dims(slice_data, axis=-1)  # [T, H, W, 1, 1]
                watermark = np.repeat(watermark, n_samples, axis=-1)  # [T, H, W, 1, S]
                id_vols.append(watermark)

            full_data = np.concatenate(id_vols + [work_data], axis=-2)
        else:
            # Point: [T, H, W, C]
            axes = ("T", "H", "W", "C")
            id_vols = [self.data[..., idx : idx + 1] for idx in primary_idxs + identity_idxs]
            full_data = np.concatenate(id_vols + [work_data], axis=-1)

        # The final channel map is [Keys] + [Identities] + [Predictions]
        final_names = primary_names + identity_names + full_signal_names

        return VolumeHandler(
            data=full_data,
            axes=axes,
            channel_map=final_names,
            time_col=time_col,
            id_col=id_col,
            spatial_cols=self._metadata.spatial_cols,
            identity_cols=tuple(identity_names),
            feature_cols=tuple(full_signal_names),
            spatial_offset=self._metadata.spatial_offset,
        )

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

        return VolumeHandler(
            data=collapsed_data,
            axes=new_axes,
            channel_map=self.channel_map,
            time_col=self._metadata.time_col,
            id_col=self._metadata.id_col,
            spatial_cols=self._metadata.spatial_cols,
            identity_cols=self._metadata.identity_cols,
            feature_cols=self._metadata.feature_cols,
            spatial_offset=self._metadata.spatial_offset,
        )

    # ─── PredictionFrame Output Path (ADR-047 pandas-free) ───────────────────

    def to_evaluation_pf(
        self,
        history: "VolumeHandler",
        start_idx: int,
        all_targets: List[str],
    ) -> Dict[str, Any]:
        """
        Converts predictions to a dict of PredictionFrames by slicing a history
        provider.  Symmetric with to_evaluation_df() — same provider-prep logic,
        no pandas object created.

        Returns
        -------
        dict[str, PredictionFrame]
            Keys are target names; value is one PredictionFrame per target
            covering this rolling-origin window.
        """
        from views_pipeline_core.data.prediction_frame import PredictionFrame  # lazy import

        duration = self.data.shape[self.get_axis_idx("T")]

        # Same bounds check as to_evaluation_df
        history_duration = history.data.shape[history.get_axis_idx("T")]
        if start_idx + duration > history_duration:
            err_msg = (
                f"VolumeHandler Contract Violation: Evaluation window "
                f"[index {start_idx} : {start_idx + duration}] "
                f"exceeds history duration ({history_duration})."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        provider_slice = history.slice_time(start_idx, start_idx + duration)
        return self._reconstruct_as_pf_dict(provider_slice, all_targets, PredictionFrame)

    def to_forecast_pf(
        self,
        history: "VolumeHandler",
        all_targets: List[str],
    ) -> Dict[str, Any]:
        """
        Converts predictions to a dict of PredictionFrames by extrapolating a
        history provider.  Symmetric with to_forecast_df() — no pandas object
        created.

        Returns
        -------
        dict[str, PredictionFrame]
            Keys are target names; value is one PredictionFrame per target.
        """
        from views_pipeline_core.data.prediction_frame import PredictionFrame  # lazy import

        duration = self._data.shape[self.get_axis_idx("T")]
        provider_future = history.extrapolate_time(duration)
        return self._reconstruct_as_pf_dict(provider_future, all_targets, PredictionFrame)

    def _valid_cell_indices(
        self, provider: "VolumeHandler"
    ) -> Tuple[np.ndarray, Tuple, np.ndarray, np.ndarray]:
        """
        Shared geometry for the PredictionFrame output path.

        Returns (temp_data, indices, time_flat, unit_flat) where:
        - temp_data  : self.data after transpose + North-Up flip  [H, W, T, C, (S)]
        - indices    : (h_arr, w_arr, t_arr) from np.where(valid_mask)
        - time_flat  : 1-D array of month_id values for each valid cell  (N,)
        - unit_flat  : 1-D array of priogrid_gid values for each valid cell  (N,)
        """
        # 1. Align self (signal)
        temp_data = (
            self._data.detach().cpu().numpy()
            if torch.is_tensor(self._data)
            else self._data.copy()
        )
        has_samples = "S" in self._metadata.axes

        if has_samples:
            t_ax, h_ax, w_ax, c_ax, s_ax = (
                self.get_axis_idx("T"), self.get_axis_idx("H"),
                self.get_axis_idx("W"), self.get_axis_idx("C"),
                self.get_axis_idx("S"),
            )
            temp_data = np.transpose(temp_data, (h_ax, w_ax, t_ax, c_ax, s_ax))
        else:
            t_ax, h_ax, w_ax, c_ax = (
                self.get_axis_idx("T"), self.get_axis_idx("H"),
                self.get_axis_idx("W"), self.get_axis_idx("C"),
            )
            temp_data = np.transpose(temp_data, (h_ax, w_ax, t_ax, c_ax))
        temp_data = np.flip(temp_data, axis=0)  # North-Up

        # 2. Align provider (scaffold)
        p_data = (
            provider.data.detach().cpu().numpy()
            if torch.is_tensor(provider.data)
            else provider.data.copy()
        )
        p_t, p_h, p_w, p_c = (
            provider.get_axis_idx("T"), provider.get_axis_idx("H"),
            provider.get_axis_idx("W"), provider.get_axis_idx("C"),
        )
        p_data = np.transpose(p_data, (p_h, p_w, p_t, p_c))
        p_data = np.flip(p_data, axis=0)  # North-Up

        # 3. Mask via priogrid_gid > 0
        id_col = provider._metadata.id_col
        time_col = provider._metadata.time_col
        pg_idx = provider.channel_map.index(id_col)
        time_idx = provider.channel_map.index(time_col)

        mask = p_data[:, :, :, pg_idx] > 0
        indices = np.where(mask)

        time_flat = p_data[indices[0], indices[1], indices[2], time_idx].astype(np.int32)
        unit_flat = p_data[indices[0], indices[1], indices[2], pg_idx].astype(np.int32)

        return temp_data, indices, time_flat, unit_flat

    def _reconstruct_as_pf_dict(
        self, provider: "VolumeHandler", all_targets: List[str], PredictionFrame: Any
    ) -> Dict[str, Any]:
        """
        Core PredictionFrame assembly.  Applies the same spatial geometry as
        _reconstruct_from_provider() but writes directly into numpy arrays —
        no Polars, no pandas, no list-in-cell overhead.
        """
        temp_data, indices, time_flat, unit_flat = self._valid_cell_indices(provider)
        has_samples = "S" in self._metadata.axes
        identifiers = {"time": time_flat, "unit": unit_flat}

        result: Dict[str, Any] = {}
        for target in all_targets:
            pred_col = f"pred_{target}"
            chan_idx = self.channel_map.index(pred_col)
            if has_samples:
                y_pred = temp_data[indices[0], indices[1], indices[2], chan_idx, :]  # (N, S)
            else:
                y_pred = temp_data[indices[0], indices[1], indices[2], chan_idx]     # (N,)
                y_pred = y_pred.reshape(-1, 1)                                        # (N, 1)
            result[target] = PredictionFrame(
                y_pred=y_pred,
                identifiers=identifiers,
            )
        return result

    # ─── DataFrame Output Path ────────────────────────────────────────────────

    def to_historical_df(self) -> pd.DataFrame:
        """
        Converts the internal volume back to a sparse DataFrame.
        """
        return self._reconstruct_from_provider(self)

    def to_evaluation_df(self, history: "VolumeHandler", start_idx: int) -> pd.DataFrame:
        """
        Converts predictions to DF by slicing a history provider.
        """
        duration = self.data.shape[self.get_axis_idx("T")]

        # Contract Validation (ADR 015)
        history_duration = history.data.shape[history.get_axis_idx("T")]
        if start_idx + duration > history_duration:
            err_msg = (
                f"VolumeHandler Contract Violation: Evaluation window "
                f"[index {start_idx} : {start_idx + duration}] "
                f"exceeds history duration ({history_duration})."
            )

            logger.error(err_msg)

            raise ValueError(err_msg)

        provider_slice = history.slice_time(start_idx, start_idx + duration)
        return self._reconstruct_from_provider(provider_slice)

    def to_forecast_df(self, history: "VolumeHandler") -> pd.DataFrame:
        """
        Converts predictions to DF by extrapolating a history provider.
        """
        duration = self._data.shape[self.get_axis_idx("T")]
        provider_future = history.extrapolate_time(duration)
        return self._reconstruct_from_provider(provider_future)

    def _reconstruct_from_provider(self, provider: "VolumeHandler") -> pd.DataFrame:
        """
        Shared logic: Align, Mask, Flatten, and Combine.
        Handles both Point (4D) and Stochastic (5D) volumes.
        Uses an Iterative Watermarked Bridge (ADR 023) to ensure
        absolute topographic integrity and RAM scalability.
        """
        # 1. Align Self (Signal)
        temp_data = (
            self._data.detach().cpu().numpy() if torch.is_tensor(self._data) else self._data.copy()
        )
        has_samples = "S" in self._metadata.axes

        if has_samples:
            t_idx, h_idx, w_idx, c_idx, s_idx = (
                self.get_axis_idx("T"),
                self.get_axis_idx("H"),
                self.get_axis_idx("W"),
                self.get_axis_idx("C"),
                self.get_axis_idx("S"),
            )
            temp_data = np.transpose(temp_data, (h_idx, w_idx, t_idx, c_idx, s_idx))
            temp_data = np.flip(temp_data, axis=0)
        else:
            t_idx, h_idx, w_idx, c_idx = (
                self.get_axis_idx("T"),
                self.get_axis_idx("H"),
                self.get_axis_idx("W"),
                self.get_axis_idx("C"),
            )
            temp_data = np.transpose(temp_data, (h_idx, w_idx, t_idx, c_idx))
            temp_data = np.flip(temp_data, axis=0)

        # 2. Align Provider (Scaffold)
        p_data = (
            provider.data.detach().cpu().numpy()
            if torch.is_tensor(provider.data)
            else provider.data.copy()
        )
        p_t, p_h, p_w, p_c = (
            provider.get_axis_idx("T"),
            provider.get_axis_idx("H"),
            provider.get_axis_idx("W"),
            provider.get_axis_idx("C"),
        )
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
            id_col: p_data[indices[0], indices[1], indices[2], pg_idx].astype(np.int32),
        }

        # Add Actuals and Identities to Scaffold from the provider's map
        for i, name in enumerate(provider.channel_map):
            if name in [time_col, id_col]:
                continue

            # Carry everything from the provider that is either an identity or a feature
            if name in provider._metadata.identity_cols or name in provider._metadata.feature_cols:
                scaffold_cols[name] = p_data[indices[0], indices[1], indices[2], i].astype(
                    np.float32
                )

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
            if name in pl_master.columns:
                continue

            # Create Watermarked Head
            if head_id_idx is not None:
                if has_samples:
                    # STOCHASTIC WATERMARK: Extract first sample only for IDs
                    # Join keys must be scalar.
                    head_dict = {
                        time_col: temp_data[
                            indices[0], indices[1], indices[2], head_time_idx, 0
                        ].astype(np.int32),
                        id_col: temp_data[
                            indices[0], indices[1], indices[2], head_id_idx, 0
                        ].astype(np.int32),
                        name: temp_data[indices[0], indices[1], indices[2], i, :],
                    }
                else:
                    # POINT WATERMARK: Standard extraction
                    head_dict = {
                        time_col: temp_data[
                            indices[0], indices[1], indices[2], head_time_idx
                        ].astype(np.int32),
                        id_col: temp_data[indices[0], indices[1], indices[2], head_id_idx].astype(
                            np.int32
                        ),
                        name: temp_data[indices[0], indices[1], indices[2], i],
                    }
            else:
                # POSITION FALLBACK: Re-use scaffold IDs (Vulnerable to shuffle)
                err_msg = (
                    f"VolumeHandler._reconstruct_from_provider: Prediction channel '{name}' "
                    "has no watermarked identity scaffold. Cannot safely reconstruct without "
                    "risking identity scramble. Ensure wrap_predictions() is used before "
                    "reconstruction."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            df_head = pl.DataFrame(head_dict)

            # Explicit Join (Red Team Proof)
            pl_master = pl_master.join(df_head, on=[time_col, id_col], how="left")
            del df_head
            gc.collect()

        # 6. Iterative Safe Handshake to Pandas (Legacy Compatibility)
        # We build the Pandas DataFrame column by column to keep the 'Object Tax'
        # limited to exactly one column's worth of Python lists.
        # ADR 032: We initialize with all columns from pl_master to ensure
        # bookkeeping identities (c_id, row, col) are carried forward.
        df_out = pd.DataFrame()

        for col in pl_master.columns:
            # Safe Export: Use to_list() for stochastic channels to ensure compatibility
            if has_samples and col in self.channel_map:
                df_out[col] = pl_master[col].to_list()
            else:
                df_out[col] = pl_master[col].to_pandas()

            gc.collect()

        # 7. Final Topographical Restoration
        if time_col in df_out.columns and id_col in df_out.columns:
            df_out = df_out.set_index([time_col, id_col])

        return df_out

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

        return VolumeHandler(
            data=new_data,
            axes=self._metadata.axes,
            channel_map=self._metadata.channel_map,
            time_col=self._metadata.time_col,
            id_col=self._metadata.id_col,
            spatial_cols=self._metadata.spatial_cols,
            identity_cols=self._metadata.identity_cols,
            feature_cols=self._metadata.feature_cols,
            spatial_offset=self._metadata.spatial_offset,
        )

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
            t_increments = torch.arange(1, steps + 1, device=self._data.device).view(
                steps, 1, 1
            )
            t_future_vol = cast(torch.Tensor, future_vol)
            t_future_vol[..., m_idx] += t_increments
        else:
            np_increments = np.arange(1, steps + 1).reshape(steps, 1, 1)
            np_future_vol = cast(np.ndarray, future_vol)
            np_future_vol[..., m_idx] += np_increments

        return VolumeHandler(
            data=future_vol,
            axes=self._metadata.axes,
            channel_map=self._metadata.channel_map,
            time_col=self._metadata.time_col,
            id_col=self._metadata.id_col,
            spatial_cols=self._metadata.spatial_cols,
            identity_cols=self._metadata.identity_cols,
            feature_cols=self._metadata.feature_cols,
            spatial_offset=self._metadata.spatial_offset,
        )

    def permute(self, dims: Union[List[int], Tuple[int, ...]]) -> "VolumeHandler":
        """
        Reorders the axes of the volume and updates the Ledger.
        NOTE: Review needed - primarily used in geometric tests.
        """
        dims_tuple = tuple(dims)
        if torch.is_tensor(self._data):
            t_data = cast(torch.Tensor, self._data)
            self._data = cast(Any, t_data.permute(*dims_tuple))
        else:
            np_data = cast(np.ndarray, self._data)
            self._data = cast(Any, np.transpose(np_data, dims_tuple))

        # Update Ledger
        new_axes = tuple(self._metadata.axes[i] for i in dims_tuple)
        self._metadata = replace(
            self._metadata,
            axes=new_axes,
            history=self._metadata.history + (("permute", dims_tuple),),
        )
        return self

    def flip(self, axis_label: str) -> "VolumeHandler":
        """
        Flips the volume along a specific named axis and updates the Ledger history.
        NOTE: Critical for data augmentation in training loop.
        """
        idx = self.get_axis_idx(axis_label)
        if torch.is_tensor(self._data):
            t_data = cast(torch.Tensor, self._data)
            self._data = cast(Any, torch.flip(t_data, dims=[idx]))
        else:
            np_data = cast(np.ndarray, self._data)
            self._data = cast(Any, np.flip(np_data, axis=idx))

        self._metadata = replace(
            self._metadata, history=self._metadata.history + (("flip", axis_label),)
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

    def _execute_derivations(self, derivations_config: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Executes additive feature engineering instructions (ADR 046).
        Mutates internal _data and _metadata.
        """
        c_idx = self.get_axis_idx("C")
        new_channels = list(self._metadata.channel_map)
        new_features = list(self._metadata.feature_cols)

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
