"""
PredictionFrameAssembler: Interface adapter between Entity-layer VolumeHandler
and Framework-layer PredictionFrame (D-01 partial split executed 2026-04-11).

Extracted from VolumeHandler to:
- Resolve C-39 (Dependency Rule): VolumeHandler no longer imports a Framework type
- Partially resolve C-36 (ISP): shrinks VolumeHandler's monolithic interface
- Partially resolve C-37 (SAP): partial abstraction at the entity/framework boundary

The assembler is stateless. Construct once per inference run or per call —
both are valid. The lazy import of the views_frames PredictionFrame leaf is
isolated to method bodies so this module imports cleanly even if views_frames
is absent (and to keep the entity/framework boundary one-directional).
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)


class PredictionFrameAssembler:
    """
    Pandas-free assembler. Converts a wrapped prediction VolumeHandler (signal)
    plus a history scaffold (provider) into a dict[str, PredictionFrame].
    """

    def assemble_evaluation(
        self,
        signal: "VolumeHandler",
        history: "VolumeHandler",
        start_idx: int,
        all_targets: List[str],
    ) -> Dict[str, Any]:
        """
        Convert predictions to a dict of PredictionFrames by slicing a history
        provider. No pandas object created.

        Parameters
        ----------
        signal : VolumeHandler
            The wrapped prediction volume produced by `wrap_predictions()` and
            (optionally) `inverse_transform_volume()` / `collapse_to_point()`.
        history : VolumeHandler
            The history scaffold providing geographic identity and time keys.
        start_idx : int
            The temporal offset within `history` where the signal starts.
        all_targets : List[str]
            Target names (the assembler looks up `pred_{target}` in
            `signal.channel_map`).

        Returns
        -------
        dict[str, PredictionFrame]
            Keys are target names; value is one PredictionFrame per target
            covering this rolling-origin window.
        """
        from views_frames import PredictionFrame  # lazy (pipeline-core 3.0.0 re-exports this leaf)

        duration = signal.data.shape[signal.get_axis_idx("T")]

        # Bounds check
        history_duration = history.data.shape[history.get_axis_idx("T")]
        if start_idx + duration > history_duration:
            err_msg = (
                f"PredictionFrameAssembler Contract Violation: Evaluation window "
                f"[index {start_idx} : {start_idx + duration}] "
                f"exceeds history duration ({history_duration})."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        provider_slice = history.slice_time(start_idx, start_idx + duration)
        return self._reconstruct_as_pf_dict(signal, provider_slice, all_targets, PredictionFrame)

    def _valid_cell_indices(
        self, signal: "VolumeHandler", provider: "VolumeHandler"
    ) -> Tuple[np.ndarray, Tuple, np.ndarray, np.ndarray]:
        """
        Shared geometry for the PredictionFrame output path.

        Returns (temp_data, indices, time_flat, unit_flat) where:
        - temp_data  : signal.data after transpose + North-Up flip  [H, W, T, C, (S)]
        - indices    : (h_arr, w_arr, t_arr) from np.where(valid_mask)
        - time_flat  : 1-D array of month_id values for each valid cell  (N,)
        - unit_flat  : 1-D array of priogrid_gid values for each valid cell  (N,)
        """
        from views_hydranet.utils.volume_handler import SpatialConvention

        if signal.spatial_convention != SpatialConvention.NORTH_UP:
            raise ValueError("Signal must be North-Up. Got: " + str(signal.spatial_convention))
        if provider.spatial_convention != SpatialConvention.NORTH_UP:
            raise ValueError("Provider must be North-Up. Got: " + str(provider.spatial_convention))

        # 1. Align signal
        # No copy: np.transpose() and np.flip() return views; fancy indexing
        # in _reconstruct_as_pf_dict() allocates only the valid (N, S) result.
        temp_data = (
            signal.data.detach().cpu().numpy() if torch.is_tensor(signal.data) else signal.data
        )
        has_samples = "S" in signal.axes

        if has_samples:
            t_ax, h_ax, w_ax, c_ax, s_ax = (
                signal.get_axis_idx("T"),
                signal.get_axis_idx("H"),
                signal.get_axis_idx("W"),
                signal.get_axis_idx("C"),
                signal.get_axis_idx("S"),
            )
            temp_data = np.transpose(temp_data, (h_ax, w_ax, t_ax, c_ax, s_ax))
        else:
            t_ax, h_ax, w_ax, c_ax = (
                signal.get_axis_idx("T"),
                signal.get_axis_idx("H"),
                signal.get_axis_idx("W"),
                signal.get_axis_idx("C"),
            )
            temp_data = np.transpose(temp_data, (h_ax, w_ax, t_ax, c_ax))
        temp_data = np.flip(temp_data, axis=0)  # North-Up

        # 2. Align provider (scaffold)
        # No copy: transpose + flip below are views; mask read is non-mutating.
        p_data = (
            provider.data.detach().cpu().numpy()
            if torch.is_tensor(provider.data)
            else provider.data
        )
        p_t, p_h, p_w, p_c = (
            provider.get_axis_idx("T"),
            provider.get_axis_idx("H"),
            provider.get_axis_idx("W"),
            provider.get_axis_idx("C"),
        )
        p_data = np.transpose(p_data, (p_h, p_w, p_t, p_c))
        p_data = np.flip(p_data, axis=0)  # North-Up

        # 3. Mask via priogrid_gid > 0
        # Use VolumeHandler's public properties (id_col, time_col) instead of
        # reaching into _metadata — small encapsulation win.
        pg_idx = provider.channel_map.index(provider.id_col)
        time_idx = provider.channel_map.index(provider.time_col)

        mask = p_data[:, :, :, pg_idx] > 0
        indices = np.where(mask)

        time_flat = p_data[indices[0], indices[1], indices[2], time_idx].astype(np.int32)
        unit_flat = p_data[indices[0], indices[1], indices[2], pg_idx].astype(np.int32)

        return temp_data, indices, time_flat, unit_flat

    def _reconstruct_as_pf_dict(
        self,
        signal: "VolumeHandler",
        provider: "VolumeHandler",
        all_targets: List[str],
        PredictionFrame: Any,
    ) -> Dict[str, Any]:
        """
        Core PredictionFrame assembly. Applies spatial geometry (transpose,
        flip, mask) then writes directly into numpy arrays — no Polars, no
        pandas, no list-in-cell overhead.
        """
        # views-frames PredictionFrame takes a typed SpatioTemporalIndex (level REQUIRED), not an
        # identifiers dict (pipeline-core #188 / 3.0.0). hydranet is grid-based → SpatialLevel.PGM.
        from views_frames import SpatialLevel, SpatioTemporalIndex

        temp_data, indices, time_flat, unit_flat = self._valid_cell_indices(signal, provider)
        has_samples = "S" in signal.axes
        # Index is identical across targets (shared time_flat/unit_flat) — build once.
        index = SpatioTemporalIndex(time=time_flat, unit=unit_flat, level=SpatialLevel.PGM)

        result: Dict[str, Any] = {}
        for target in all_targets:
            pred_col = f"pred_{target}"
            chan_idx = signal.channel_map.index(pred_col)
            if has_samples:
                y_pred = temp_data[indices[0], indices[1], indices[2], chan_idx, :]  # (N, S)
            else:
                y_pred = temp_data[indices[0], indices[1], indices[2], chan_idx]  # (N,)
                y_pred = y_pred.reshape(-1, 1)  # (N, 1)
            result[target] = PredictionFrame(y_pred=y_pred, index=index)
        return result
