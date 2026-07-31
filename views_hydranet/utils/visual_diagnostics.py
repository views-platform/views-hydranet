"""
VisualDiagnostics: The Visual Truth Engine.
Governed by the Hybrid Diagnostics Plan (2026-02-19).
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from views_hydranet.utils.training_forensics import THETA_COLLAPSE_COV, THETA_HEALTH_COV
from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)

# issue #215: matplotlib is imported LAZILY via _load_mpl() — NEVER at module level. The model-run
# path imports this module (manager/inference/training), but VisualDiagnostics is a Null Object
# (diagnostics default off) → a headless run needs no plotting stack. matplotlib is an optional
# 'viz' extra, imported only when a plotting method runs.
_plt = None
_patches = None


def _load_mpl():
    """Lazy, memoized matplotlib (Agg backend) -> ``(plt, patches)``. Imported only when a plotting
    method runs (diagnostics active). Agg is set BEFORE pyplot import (headless; prevents the
    tkinter-thread crash — test_falsification_matplotlib_backend::test_P1)."""
    global _plt, _patches
    if _plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        _plt, _patches = plt, patches
    return _plt, _patches


class VisualDiagnostics:
    """
    A unified observer for generating 'Visual Biopsies' of spatiotemporal data
    at every stage of the pipeline.

    Implements the 'Null Object' pattern if diagnostic_visualizations is False.
    """

    def __init__(self, config: Dict[str, Any], run_timestamp: Optional[str] = None) -> None:
        self.active = config.get("diagnostic_visualizations", False)
        self.config = config  # Store for dynamic labelling

        if self.active:
            # Determine save directory
            # If manager passed a timestamp in config (ADR 026), use it.
            from datetime import datetime

            config_ts = config.get("model_time_stamp")
            ts = run_timestamp or config_ts or datetime.now().strftime("%d%m%y_%H%M")
            self.save_dir = f"reports/plots/diagnostics/{ts}"

            # ADR 045: Forensic Archive Structure
            self.dirs = {
                "pipeline": os.path.join(self.save_dir, "01_pipeline_health"),
                "training": os.path.join(self.save_dir, "02_training_dynamics"),
                "reasoning": os.path.join(self.save_dir, "03_model_reasoning"),
            }
            for d in self.dirs.values():
                os.makedirs(d, exist_ok=True)

            logger.info(f"📸 VisualDiagnostics: Active. Saving biopsies to {self.save_dir}")

            # Cache spatial columns for dataframe reconstruction
            self.y_col, self.x_col = config.get("spatial_cols", ["row_col", "col_col"])
            self.time_col = config.get("time_col", "month_id")
            self.height = config.get("height", 180)
            self.width = config.get("width", 360)
            self.row_offset = config.get("row_offset", 0)
            self.col_offset = config.get("col_offset", 0)
        else:
            logger.debug("VisualDiagnostics: Inactive.")

    def biopsy_dataframe(
        self, df: pd.DataFrame, stage_label: str, features: List[str] = []
    ) -> None:
        """
        Adapts a flat or MultiIndexed DataFrame to the biopsy grid.
        Routes to '01_pipeline_health' (ADR 045).
        """
        if not self.active:
            return

        try:
            # 0. Handle MultiIndex (Crucial for Stage 1)
            # If the dataframe is indexed, reset it so we can treat everything as columns
            working_df = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df.copy()

            # 1. Sort for temporal continuity (Hypothesis 5 Probe)
            df_sorted = working_df.sort_values(by=[self.time_col, self.y_col, self.x_col])

            # 2. Reshape to [T, H, W, C] using VolumeHandler logic (but lightweight)
            t_min = df_sorted[self.time_col].min()
            t_max = df_sorted[self.time_col].max()
            t_len = int(t_max - t_min + 1)

            # Select 5 timestamps
            t_indices = np.linspace(0, t_len - 1, 5, dtype=int)
            global_months = t_min + t_indices

            # Filter for these months only to save memory
            df_slice = df_sorted[df_sorted[self.time_col].isin(global_months)]

            if df_slice.empty:
                logger.error(
                    "VisualDiagnostics: no data for selected timestamps — skipping '%s'.",
                    stage_label,
                )
                return

            # We need to construct a mini-volume for plotting
            vol_slice = np.zeros((5, self.height, self.width, len(features)))
            vol_slice[:] = np.nan

            r_idx = (df_slice[self.y_col] - self.row_offset).astype(int).values
            c_idx = (df_slice[self.x_col] - self.col_offset).astype(int).values

            for label, idx_arr, limit, col, offset in [
                ("row", r_idx, self.height, self.y_col, self.row_offset),
                ("col", c_idx, self.width, self.x_col, self.col_offset),
            ]:
                if idx_arr.min() < 0:
                    logger.error(
                        "VisualDiagnostics: %s indices negative after offset. "
                        "min %s=%s, %s_offset=%s — skipping '%s'.",
                        label,
                        label,
                        df_slice[col].min(),
                        label,
                        offset,
                        stage_label,
                    )
                    return
                if idx_arr.max() >= limit:
                    logger.error(
                        "VisualDiagnostics: %s index %d >= %s %d. "
                        "max %s=%s, %s_offset=%s — skipping '%s'.",
                        label,
                        idx_arr.max(),
                        label,
                        limit,
                        label,
                        df_slice[col].max(),
                        label,
                        offset,
                        stage_label,
                    )
                    return

            month_map = {m: idx for idx, m in enumerate(global_months)}
            t_idx = df_slice[self.time_col].map(month_map).values

            for i, feat in enumerate(features):
                if feat not in df_slice.columns:
                    continue

                raw_vals = df_slice[feat].values
                first_valid = next((v for v in raw_vals if v is not None), None)
                if isinstance(first_valid, (list, np.ndarray)):
                    scalar_vals = np.array([np.mean(v) for v in raw_vals], dtype=float)
                else:
                    scalar_vals = raw_vals.astype(float)

                vol_slice[t_idx, r_idx, c_idx, i] = scalar_vals

            # 3. Orient North-Up (Match VolumeHandler logic)
            # Fancy indexing places min-latitude at index 0.
            # We flip axis 1 (Height) so North is at index 0 for 'upper' plotting.
            vol_slice = np.flip(vol_slice, axis=1)

            # Plot
            self._plot_grid(
                vol_slice, features, global_months, stage_label, subdir=self.dirs["pipeline"]
            )

        except Exception:
            logger.error(
                "VisualDiagnostics: biopsy_dataframe failed at '%s' — skipping plot.",
                stage_label,
                exc_info=True,
            )

    def biopsy_volume(self, vh: VolumeHandler, stage_label: str) -> None:
        """
        Adapts a VolumeHandler to the biopsy grid.
        This is the most trusted view because it reflects the internal array state.
        """
        if not self.active:
            return

        try:
            # 1. Extract Data [T, H, W, C]
            data = vh.data.detach().cpu().numpy() if torch.is_tensor(vh.data) else vh.data

            # Handle Stochastic [T, H, W, C, S] -> Mean -> [T, H, W, C]
            if "S" in vh.axes:
                s_idx = vh.get_axis_idx("S")
                data = np.nanmean(data, axis=s_idx)

            # Ensure T, H, W, C order
            t_idx, h_idx, w_idx, c_idx = (
                vh.get_axis_idx("T"),
                vh.get_axis_idx("H"),
                vh.get_axis_idx("W"),
                vh.get_axis_idx("C"),
            )
            data = np.transpose(data, (t_idx, h_idx, w_idx, c_idx))

            # Select 5 timestamps
            t_len = data.shape[0]
            t_indices = np.linspace(0, t_len - 1, 5, dtype=int)

            # Select Features (Metadata first, then signals)
            interesting, _ = self._select_display_channels(vh)

            # Construct Sliced Volume [5, H, W, F]
            feat_indices = [vh.channel_map.index(c) for c in interesting]
            vol_slice = data[t_indices][..., feat_indices]

            # ADR 045: Route predicted volumes to Reasoning, others to Pipeline
            target_subdir = (
                self.dirs["reasoning"]
                if "predict" in stage_label.lower()
                else self.dirs["pipeline"]
            )

            self._plot_grid(vol_slice, interesting, t_indices, stage_label, subdir=target_subdir)

        except Exception:
            logger.error(
                "VisualDiagnostics: biopsy_volume failed at '%s' — skipping plot.",
                stage_label,
                exc_info=True,
            )

    def biopsy_tensor(
        self, tensor: torch.Tensor, stage_label: str, channel_names: List[str]
    ) -> None:
        """
        Adapts a PyTorch Tensor [B, T, C, H, W] to the biopsy grid.
        Assumes Batch=0.
        """
        if not self.active:
            return

        try:
            # [B, T, C, H, W] -> [T, H, W, C]
            if tensor.ndim == 5:
                data = tensor[0].permute(0, 2, 3, 1).detach().cpu().numpy()
            elif tensor.ndim == 4:  # [T, C, H, W]
                data = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
            else:
                logger.warning(f"VisualDiagnostics: Tensor shape {tensor.shape} not supported.")
                return

            t_len = data.shape[0]
            t_indices = np.linspace(0, t_len - 1, 5, dtype=int)

            # Slice time
            data = data[t_indices]  # [5, H, W, C]

            self._plot_grid(
                data, channel_names, t_indices, stage_label, subdir=self.dirs["pipeline"]
            )

        except Exception:
            logger.error(
                "VisualDiagnostics: biopsy_tensor failed at '%s' — skipping plot.",
                stage_label,
                exc_info=True,
            )

    def biopsy_sample(
        self, sample_vh: VolumeHandler, global_vh: VolumeHandler, stage_label: str
    ) -> None:
        """
        Specialized biopsy for training samples.
        Shows the local patch alongside its global geographic context.
        """
        if not self.active:
            return

        try:
            # 1. Select interesting features (Metadata first, then signals)
            interesting, meta_cols = self._select_display_channels(sample_vh)

            # 2. Extract Sample Data [5, H_patch, W_patch, F]
            s_data = sample_vh.data
            t_idx, h_idx, w_idx, c_idx = (
                sample_vh.get_axis_idx("T"),
                sample_vh.get_axis_idx("H"),
                sample_vh.get_axis_idx("W"),
                sample_vh.get_axis_idx("C"),
            )
            s_data = np.transpose(s_data, (t_idx, h_idx, w_idx, c_idx))

            t_len = s_data.shape[0]
            t_indices = np.linspace(0, t_len - 1, 5, dtype=int)
            feat_indices = [sample_vh.channel_map.index(c) for c in interesting]
            sample_slice = s_data[t_indices][..., feat_indices]

            # 3. Extract Global Context Sequence for the signal feature
            # We want to see how the whole world evolves alongside the patch
            g_data = global_vh.data
            g_t_idx = global_vh.get_axis_idx("T")
            g_c_idx = global_vh.get_axis_idx("C")

            signal_feat = next((c for c in interesting if c not in meta_cols), None)
            if signal_feat is None:
                logger.warning(
                    f"VisualDiagnostics.biopsy_sample: No signal feature found in channel "
                    f"map for global context at {stage_label}. Skipping biopsy."
                )
                return
            g_feat_idx = global_vh.channel_map.index(signal_feat)
            id_feat_idx = global_vh.channel_map.index(global_vh.id_col)

            # Extract [5, H, W] global maps
            # We need to handle axis permutation safely
            global_maps = []
            for t in t_indices:
                slc = [slice(None)] * g_data.ndim
                slc[g_t_idx] = t
                slc[g_c_idx] = g_feat_idx
                g_map = g_data[tuple(slc)]

                slc[g_c_idx] = id_feat_idx
                id_map = g_data[tuple(slc)]

                # Mask Ocean
                masked_map = np.where(id_map > 0, g_map, np.nan)
                global_maps.append(masked_map)

            global_maps = np.stack(global_maps)

            # 4. Calculate Relative Offset for Bounding Box
            # sample_vh.spatial_offset = (global_row_offset + r0, global_col_offset + c0)
            # relative_r0 = sample_vh.spatial_offset[0] - global_vh.spatial_offset[0]
            # relative_c0 = sample_vh.spatial_offset[1] - global_vh.spatial_offset[1]
            rel_offset = (
                sample_vh.spatial_offset[0] - global_vh.spatial_offset[0],
                sample_vh.spatial_offset[1] - global_vh.spatial_offset[1],
            )

            # 5. Plot with Context
            self._plot_grid_with_context(
                sample_slice,
                interesting,
                t_indices,
                global_maps,
                signal_feat,
                rel_offset,
                stage_label,
                subdir=self.dirs["training"],
            )

        except Exception:
            logger.error(
                "VisualDiagnostics: biopsy_sample failed at '%s' — skipping plot.",
                stage_label,
                exc_info=True,
            )

    def biopsy_health_constellation(
        self, weight_norms: Dict[str, float], stage_label: str
    ) -> None:
        """
        Produces a radar (polar) plot of mean L2 weight norms per functional block.
        Implements ADR-037: Geometric Health Visualization (Health Constellations).
        Routes to '02_training_dynamics/' (ADR-045).
        Saved as constellation_{safe_label}.png.
        """
        if not self.active:
            return

        plt, _ = _load_mpl()  # #215: lazy matplotlib, only when active
        try:
            # 1. Aggregate raw layer norms into functional blocks
            blocks: Dict[str, List[float]] = {
                "Encoder": [],
                "Bottleneck": [],
                "Decoder": [],
                "MultiTaskHead": [],
            }
            # Map layer names → functional blocks. The model uses short prefixes (enc_/bottleneck_/
            # dec_/upsample_) and fuses the decoder into per-head paths (dec_conv*_head*), with the
            # final conv (dec_conv4_head*) as the output head. Match those, NOT literal "encoder"/
            # "decoder" (which never appear → previously left Encoder & Decoder permanently empty,
            # rendering a degenerate vertical line).
            for name, norm in weight_norms.items():
                short = name.replace("module.", "").lower()
                if "bottleneck" in short:
                    blocks["Bottleneck"].append(norm)
                elif "enc" in short:
                    blocks["Encoder"].append(norm)
                elif "conv4" in short:  # final per-head output conv = the prediction heads
                    blocks["MultiTaskHead"].append(norm)
                elif "dec" in short or "upsample" in short:
                    blocks["Decoder"].append(norm)

            labels = list(blocks.keys())
            values = [float(np.mean(v)) if v else 0.0 for v in blocks.values()]

            # 2. Close the radar polygon
            n = len(labels)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
            angles += angles[:1]
            values += values[:1]

            # 3. Plot
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
            ax.plot(angles, values, "o-", linewidth=2, color="cyan")
            ax.fill(angles, values, alpha=0.25, color="cyan")
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_title(f"Health Constellation: {stage_label}", fontsize=14, pad=20)

            # 4. Save
            safe_label = stage_label.lower().replace(" ", "_").replace("/", "_")
            save_path = os.path.join(self.dirs["training"], f"constellation_{safe_label}.png")
            plt.savefig(save_path, dpi=100)
            plt.close()
            logger.info(f"📸 VisualDiagnostics: Saved {save_path}")

        except Exception:
            logger.error(
                "VisualDiagnostics: biopsy_health_constellation failed at '%s' — skipping plot.",
                stage_label,
                exc_info=True,
            )

    def _select_display_channels(self, vh: VolumeHandler) -> Tuple[List[str], set]:
        """
        Builds the ordered channel list for biopsy display.

        Returns (interesting, meta_cols) where:
        - interesting: ordered list of channel names to display (metadata first, then signals)
        - meta_cols: set of identity/metadata channel names. Used by biopsy_sample to
          distinguish 'context' (signal) from 'location' (metadata).
        """
        meta_cols = {vh.time_col, vh.id_col, "c_id"} | set(vh.spatial_cols)
        interesting: List[str] = []

        # 1. Metadata / Identity Group (fixed display order)
        meta_order = [vh.time_col, vh.id_col, "c_id"] + list(vh.spatial_cols)
        for c in meta_order:
            if c in vh.channel_map and c not in interesting:
                interesting.append(c)

        # 2. Signal Features Group (pulled from ledger — ADR 012)
        for c in vh.feature_cols:
            if c in vh.channel_map and c not in interesting:
                interesting.append(c)

        return interesting, meta_cols

    def biopsy_autoregressive(
        self,
        truth_seq: List[np.ndarray],
        pred_seq: List[np.ndarray],
        stage_label: str,
        channel_names: List[str],
        time_indices: Optional[List[float]] = None,
    ) -> None:
        """
        Specialized biopsy for the autoregressive feedback loop.
        Structure: 3 Rows (Truth, Pred, Delta) x 6 Columns (Seed + 5 steps).
        """
        if not self.active:
            return

        plt, _ = _load_mpl()  # #215: lazy matplotlib, only when active
        try:
            # Inputs are lists of arrays [H, W, C]
            # Convert to [Time, Row, Col, Chan]
            truth = np.stack(truth_seq)  # [6, H, W, C]
            pred = np.stack(pred_seq)  # [6, H, W, C]
            delta = np.abs(truth - pred)

            # Plot the first signal only (Joyful). truth/pred/delta are [T, H, W, C] over the
            # full rollout horizon (the caller passes the whole accumulator, padded to >=6).
            feat_idx = 0
            feat_name = channel_names[feat_idx]
            T = truth.shape[0]

            # Horizon columns: t=0 / early / late rollout snapshots + the full-rollout TEMPORAL
            # MEAN. Shows t=0 sharp/calibrated while deep-rollout steps inflate & centralize (the
            # inflation hidden in any single-step view); 'full' = the aggregate over the horizon.
            def _mlabel(idx):
                return (
                    f" m{int(time_indices[idx])}"
                    if time_indices and idx < len(time_indices)
                    else ""
                )

            i_early, i_late = T // 3, T - 1
            cols = [
                (f"t=0{_mlabel(0)}", truth[0], pred[0], delta[0]),
                (f"early{_mlabel(i_early)}", truth[i_early], pred[i_early], delta[i_early]),
                (f"late{_mlabel(i_late)}", truth[i_late], pred[i_late], delta[i_late]),
                ("full (mean)", truth.mean(0), pred.mean(0), delta.mean(0)),
            ]
            n_cols = len(cols)
            fig, axes = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 10), squeeze=False)

            row_labels = ["GROUND TRUTH (y)", "PREDICTION (ŷ)", "ABSOLUTE DELTA (|y-ŷ|)"]
            # Shared scale for Truth and Pred across all columns; Delta gets its own.
            v_min = float(
                np.min([np.nanmin(truth[..., feat_idx]), np.nanmin(pred[..., feat_idx])])
            )
            v_max = float(
                np.max([np.nanmax(truth[..., feat_idx]), np.nanmax(pred[..., feat_idx])])
            )
            d_max = float(np.nanmax(delta[..., feat_idx]))

            for ci, (ctitle, t_img, p_img, d_img) in enumerate(cols):
                row_imgs = [t_img[..., feat_idx], p_img[..., feat_idx], d_img[..., feat_idx]]
                for r_idx in range(3):
                    ax = axes[r_idx][ci]
                    cmap = "magma" if r_idx < 2 else "Reds"
                    vmx = v_max if r_idx < 2 else d_max
                    vmn = v_min if r_idx < 2 else 0
                    ax.imshow(
                        row_imgs[r_idx],
                        origin="upper",
                        cmap=cmap,
                        vmin=vmn,
                        vmax=vmx,
                        interpolation="nearest",
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if r_idx == 0:
                        ax.set_title(ctitle, fontweight="bold")
                    if ci == 0:
                        ax.set_ylabel(
                            row_labels[r_idx], rotation=0, labelpad=80, fontweight="bold"
                        )

            plt.suptitle(
                f"Autoregressive Forensic: {stage_label} ({feat_name}) — horizon split "
                f"[{T} steps]",
                fontsize=16,
                y=0.98,
            )
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))

            # Cyan delimiter before the 'full (mean)' column (snapshots vs the aggregate).
            fig.canvas.draw()
            pos2 = axes[0][n_cols - 2].get_position()
            pos3 = axes[0][n_cols - 1].get_position()
            line_x = (pos2.x1 + pos3.x0) / 2
            fig.add_artist(
                plt.Line2D(
                    [line_x, line_x],
                    [0.05, 0.9],
                    color="cyan",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                )
            )

            # Sanitize label for filesystem
            safe_label = stage_label.lower().replace(" ", "_").replace("/", "_")
            save_path = os.path.join(self.dirs["reasoning"], f"biopsy_{safe_label}.png")
            plt.savefig(save_path, dpi=100)
            plt.close()
            logger.info(f"📸 VisualDiagnostics: Saved {save_path}")

        except Exception:
            logger.error(
                "VisualDiagnostics: biopsy_autoregressive failed at '%s' — skipping plot.",
                stage_label,
                exc_info=True,
            )

    def biopsy_training_performance(
        self,
        y_reg: np.ndarray,
        y_hat_reg: np.ndarray,
        y_cls: np.ndarray,
        y_hat_cls: np.ndarray,
        stage_label: str,
        time_indices: Optional[List[float]] = None,
        self_zeroed: bool = False,
    ) -> None:
        """
        5xN Forensic Grid for Training runs.
        Rows: [truth Reg, body E[y], truth Cls, cls gate, THE FORECAST].
        The last row is the ACTUAL scored forecast, which differs by mode:
        - ``self_zeroed=True`` (ZINB): forecast = the self-zeroed body ``y_hat_reg`` (NO ×gate;
          multiplying would double-count the structural zeros).
        - ``self_zeroed=False`` (NB=gated_NB, and legacy hurdle): forecast = ``gate × body``.
        Cols: up to 6 sequential time steps.
        """
        if not self.active:
            return

        plt, _ = _load_mpl()  # #215: lazy matplotlib, only when active
        try:
            # Inputs are [T, H, W, C]
            n_times = min(6, y_reg.shape[0])

            # 5th row = THE ACTUAL FORECAST, honest per mode (never double-gate self-zeroed).
            # - self_zeroed (ZINB): forecast IS the self-zeroed body y_hat_reg=(1-π)μ. Row 2 and
            #   row 5 coincide by design — for ZINB the body IS the forecast (no separate gate).
            # - gated (NB=gated_NB, legacy hurdle): forecast = gate × body — the diffuse body
            #   (row 2) sharpened by the classification gate (row 4).
            if self_zeroed:
                y_hat_forecast = y_hat_reg
                forecast_label = "FORECAST (self-zeroed E[y])"
            else:
                y_hat_forecast = y_hat_cls * y_hat_reg
                forecast_label = "FORECAST (gated: gate·body)"

            fig, axes = plt.subplots(5, n_times, figsize=(18, 15))

            row_labels = [
                "GROUND TRUTH (Reg)",
                "PREDICTION: body E[y]",
                "GROUND TRUTH (Cls)",
                "PREDICTION: cls gate",
                forecast_label,
            ]
            data_rows = [y_reg, y_hat_reg, y_cls, y_hat_cls, y_hat_forecast]
            cmaps = ["magma", "magma", "viridis", "viridis", "magma"]

            for r_idx in range(5):
                row_data = data_rows[r_idx]
                feat_slice = row_data[..., 0]  # Use first target only
                vmin, vmax = np.nanmin(feat_slice), np.nanmax(feat_slice)

                for t_idx in range(n_times):
                    ax = axes[r_idx, t_idx]
                    img = feat_slice[t_idx]
                    stats = self._calculate_stats(img)

                    ax.imshow(
                        img,
                        origin="upper",
                        cmap=cmaps[r_idx],
                        vmin=vmin,
                        vmax=vmax,
                        interpolation="nearest",
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if r_idx == 0:
                        m_id = int(time_indices[t_idx]) if time_indices else t_idx
                        # Midpoint split: 3 in, 3 out
                        suffix = "_in" if t_idx < 3 else "_out"
                        ax.set_title(f"{m_id}{suffix}", fontweight="bold")
                    if t_idx == 0:
                        ax.set_ylabel(
                            f"{row_labels[r_idx]}\n{stats}",
                            rotation=0,
                            labelpad=80,
                            fontsize=9,
                            fontweight="bold",
                        )
                    else:
                        ax.text(
                            0.05,
                            0.95,
                            stats,
                            transform=ax.transAxes,
                            color="white",
                            fontsize=8,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
                        )

            plt.suptitle(f"Training Performance Forensic: {stage_label}", fontsize=18, y=0.98)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))

            # Vertical line between History (3) and Forecast (3)
            if n_times > 3:
                fig.canvas.draw()
                pos2 = axes[0, 2].get_position()
                pos3 = axes[0, 3].get_position()
                line_x = (pos2.x1 + pos3.x0) / 2
                fig.add_artist(
                    plt.Line2D(
                        [line_x, line_x],
                        [0.05, 0.9],
                        color="cyan",
                        linestyle="--",
                        linewidth=2,
                        alpha=0.8,
                    )
                )

            safe_label = stage_label.lower().replace(" ", "_").replace("/", "_")
            save_path = os.path.join(self.dirs["training"], f"biopsy_{safe_label}.png")
            plt.savefig(save_path, dpi=100)
            plt.close()
            logger.info(f"📸 VisualDiagnostics: Saved {save_path}")

        except Exception:
            logger.error(
                "VisualDiagnostics: biopsy_training_performance failed at '%s' — skipping plot.",
                stage_label,
                exc_info=True,
            )

    def biopsy_loss_curves(
        self,
        history_reg: List[float],
        history_cls: List[float],
        history_total: List[float],
        stage_label: str,
    ) -> None:
        """
        Generates both Linear and Log-scale plots showing loss evolution over lessons.
        """
        if not self.active:
            return

        plt, _ = _load_mpl()  # #215: lazy matplotlib (bound before the nested closure below)

        def _generate_plot(is_log: bool):
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))

            l_reg = self.config.get("loss_reg", "Unknown")
            l_cls = self.config.get("loss_class", "Unknown")

            titles = [
                f"Regression Loss ({l_reg})",
                f"Classification Loss ({l_cls})",
                "Total Multi-Task Loss",
            ]
            data = [history_reg, history_cls, history_total]
            colors = ["firebrick", "seagreen", "royalblue"]

            for i in range(3):
                ax = axes[i]
                # Filter out non-positive for log plot to avoid matplotlib warnings
                plot_data = data[i]
                ax.plot(
                    plot_data, marker="o", linestyle="-", color=colors[i], markersize=4, alpha=0.7
                )

                scale_suffix = " (Log Scale)" if is_log else ""
                ax.set_title(titles[i] + scale_suffix, fontweight="bold")
                ax.set_xlabel("Lesson")
                ax.set_ylabel("Loss")
                ax.grid(True, which="both", alpha=0.3)

                if is_log:
                    ax.set_yscale("log")

                if len(plot_data) > 0:
                    current = plot_data[-1]
                    ax.text(
                        0.95,
                        0.95,
                        f"Current: {current:.4f}",
                        transform=ax.transAxes,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

            plt.suptitle(f"Learning Dynamics: {stage_label}", fontsize=18)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))

            fname = "loss_evolution_log.png" if is_log else "loss_evolution.png"
            save_path = os.path.join(self.dirs["training"], fname)
            plt.savefig(save_path, dpi=100)
            plt.close()

        try:
            _generate_plot(is_log=False)
            _generate_plot(is_log=True)
        except Exception:
            logger.error(
                "VisualDiagnostics: biopsy_loss_curves failed — skipping plot.",
                exc_info=True,
            )

    def biopsy_feature_dossier(
        self,
        target_name: str,
        dossier: Dict[str, List[float]],
        stage_label: str,
        target_type: str = "REG",
    ) -> None:
        """
        Generates Joyful Forensic Dossiers.
        One-metric-per-row layout.
        """
        if not self.active:
            return

        plt, _ = _load_mpl()  # #215: lazy matplotlib, only when active
        is_reg = target_type == "REG"
        metrics = (
            self.config.get("regression_metrics", [])
            if is_reg
            else self.config.get("classification_metrics", [])
        )

        # Filter for metrics actually in the dossier
        active_metrics = [m for m in metrics if m in dossier]

        # Determine Rows: N metrics + Magnitude + Bias (+ family parameter-health rows, C-213).
        # Param-health rows appear ONLY when TrainingForensics recorded family params (nb/zinb):
        # μ̄ + θ-CoV always, + π (mean/range) for zinb. Non-family dossiers lack the keys → same.
        n_metrics = len(active_metrics)
        has_param_health = is_reg and "theta_cov" in dossier
        has_pi = has_param_health and "pi_bar" in dossier
        n_param_health = 0 if not has_param_health else (3 if has_pi else 2)
        n_extra = (
            2 if is_reg else 1
        ) + n_param_health  # Reg gets Mag+Bias(+param-health), Cls Bias
        total_rows = n_metrics + n_extra

        try:
            # Horizon-split columns: t=0 (first forecast step) / early / late rollout vs the
            # full-window aggregate. Shows t=0 is calibrated while later steps inflate (hidden in
            # the full-window average). Per-slice series come from TrainingForensics (h-slices).
            cols = [
                ("t0", "t=0 (step-1)"),
                ("early", "early rollout"),
                ("late", "late rollout"),
                ("full", "full window"),
            ]
            n_cols = len(cols)
            fig, axes = plt.subplots(
                total_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * total_rows), squeeze=False
            )

            def ck(stat, ckey):
                # 'full' uses the un-prefixed key; slices use the 't0_'/'early_'/'late_' prefix.
                return stat if ckey == "full" else f"{ckey}_{stat}"

            def shared_hi(series_list):
                vals = [v for s in series_list for v in s]
                return max(vals) if vals else 1.0

            # 1. Metric rows (one per active metric), shared y per row across the 4 columns.
            for r, m in enumerate(active_metrics):
                hi = shared_hi([dossier.get(ck(m, c), []) for c, _ in cols])
                for ci, (ckey, clabel) in enumerate(cols):
                    ax = axes[r][ci]
                    ax.plot(dossier.get(ck(m, ckey), []), marker="o", color="royalblue", alpha=0.7)
                    ax.grid(True, alpha=0.3)
                    if not is_reg:
                        ax.set_ylim(0, 1.05)
                    elif hi > 0:
                        ax.set_ylim(0, hi * 1.1)
                    if r == 0:
                        ax.set_title(clabel, fontweight="bold")
                    if ci == 0:
                        ax.set_ylabel(f"Metric: {m.upper()}", fontweight="bold")

            # 2. Magnitude row (reg only): y_bar (actual) vs ŷ_bar (pred), shared scale.
            if is_reg:
                r = n_metrics
                hi = shared_hi(
                    [dossier.get(ck("y_bar", c), []) for c, _ in cols]
                    + [dossier.get(ck("y_hat_bar", c), []) for c, _ in cols]
                )
                for ci, (ckey, clabel) in enumerate(cols):
                    ax = axes[r][ci]
                    ax.plot(
                        dossier.get(ck("y_bar", ckey), []),
                        color="black",
                        linewidth=2,
                        label="y_bar",
                    )
                    ax.plot(
                        dossier.get(ck("y_hat_bar", ckey), []),
                        color="orange",
                        linestyle="--",
                        alpha=0.85,
                        label="ŷ_bar",
                    )
                    if hi > 0:
                        ax.set_ylim(0, hi * 1.1)
                    ax.grid(True, alpha=0.3)
                    if r == 0:
                        ax.set_title(clabel, fontweight="bold")
                    if ci == 0:
                        ax.set_ylabel("Magnitude Pulse\n(y_bar vs ŷ_bar)", fontweight="bold")
                        ax.legend(fontsize=7)

            # 3. Calibration row (ŷ_bar/y_bar ratio). full = instant+running; slices = one ratio.
            # Always log-scaled with FIXED bounds so this row is directly comparable across runs
            # (e.g. a pos_weight dial sweep) — no per-run linear↔log switch that breaks comparison.
            r = n_metrics + (1 if is_reg else 0)
            cal_lo, cal_hi = (0.5, 2000.0) if is_reg else (0.1, 200.0)
            for ci, (ckey, clabel) in enumerate(cols):
                ax = axes[r][ci]
                if ckey == "full":
                    ax.plot(
                        dossier.get("bias_instant", []),
                        color="firebrick",
                        alpha=0.6,
                        label="instant",
                    )
                    ax.plot(
                        dossier.get("bias_running", []),
                        color="royalblue",
                        linewidth=2,
                        label="running",
                    )
                else:
                    ax.plot(
                        dossier.get(f"{ckey}_bias", []),
                        color="firebrick",
                        alpha=0.8,
                        label="instant",
                    )
                ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
                ax.set_yscale("log")
                ax.set_ylim(cal_lo, cal_hi)
                ax.grid(True, alpha=0.3)
                if r == 0:
                    ax.set_title(clabel, fontweight="bold")
                if ci == 0:
                    lbl = (
                        "Calibration Pulse\n(ŷ_bar/y_bar)"
                        if is_reg
                        else "Detection Bias\n(ŷ/y events)"
                    )
                    ax.set_ylabel(lbl, fontweight="bold")
                    ax.legend(fontsize=7)

            # 4. Parameter-health rows (family heads only, C-213): μ̄ (conditional magnitude),
            # θ cross-cell CoV with the F1 guide lines (health >0.10, collapse <0.02), and — for
            # zinb — π mean + [min,max] range (degeneracy → 0/1). Turns the pre-registered F1
            # falsifier into a live per-lesson trace; renders for ALL targets (not just target-0).
            if has_param_health:
                base = n_metrics + 2  # after Magnitude (n_metrics) + Calibration (n_metrics+1)

                # μ̄ trajectory
                hi = shared_hi([dossier.get(ck("mu_bar", c), []) for c, _ in cols])
                for ci, (ckey, _clabel) in enumerate(cols):
                    ax = axes[base][ci]
                    ax.plot(
                        dossier.get(ck("mu_bar", ckey), []),
                        color="darkgreen",
                        marker="o",
                        alpha=0.7,
                    )
                    if hi > 0:
                        ax.set_ylim(0, hi * 1.1)
                    ax.grid(True, alpha=0.3)
                    if ci == 0:
                        ax.set_ylabel("μ̄ (body mean)\nconditional magnitude", fontweight="bold")

                # θ cross-cell CoV (std/mean over active cells) with the F1 guide lines
                hi = max(
                    shared_hi([dossier.get(ck("theta_cov", c), []) for c, _ in cols]),
                    THETA_HEALTH_COV * 1.5,
                )
                for ci, (ckey, _clabel) in enumerate(cols):
                    ax = axes[base + 1][ci]
                    ax.plot(
                        dossier.get(ck("theta_cov", ckey), []),
                        color="purple",
                        marker="o",
                        alpha=0.7,
                    )
                    ax.axhline(
                        THETA_HEALTH_COV,
                        color="green",
                        linestyle="--",
                        alpha=0.6,
                        label="health >0.10",
                    )
                    ax.axhline(
                        THETA_COLLAPSE_COV,
                        color="red",
                        linestyle="--",
                        alpha=0.6,
                        label="collapse <0.02",
                    )
                    ax.set_ylim(0, hi * 1.1)
                    ax.grid(True, alpha=0.3)
                    if ci == 0:
                        ax.set_ylabel("θ CoV (std/mean)\nheteroscedasticity", fontweight="bold")
                        ax.legend(fontsize=6)

                # π mean + [min, max] band (zinb only) — degeneracy watch (→0 plain-NB, →1 dead)
                if has_pi:
                    for ci, (ckey, _clabel) in enumerate(cols):
                        ax = axes[base + 2][ci]
                        pbar = dossier.get(ck("pi_bar", ckey), [])
                        pmin = dossier.get(ck("pi_min", ckey), [])
                        pmax = dossier.get(ck("pi_max", ckey), [])
                        ax.plot(pbar, color="teal", marker="o", alpha=0.8, label="π̄")
                        if pbar and len(pmin) == len(pbar) == len(pmax):
                            ax.fill_between(
                                range(len(pbar)),
                                pmin,
                                pmax,
                                color="teal",
                                alpha=0.2,
                                label="[min,max]",
                            )
                        ax.set_ylim(-0.02, 1.02)
                        ax.grid(True, alpha=0.3)
                        if ci == 0:
                            ax.set_ylabel(
                                "π (structural zero)\nmean + [min,max]", fontweight="bold"
                            )
                            ax.legend(fontsize=6)

            mode_str = "REGRESSION" if is_reg else "CLASSIFICATION"
            plt.suptitle(
                f"{mode_str} FORENSIC: {target_name} ({stage_label}) — horizon split", fontsize=16
            )
            plt.tight_layout(rect=(0, 0.02, 1, 0.97))

            type_tag = "reg" if is_reg else "cls"
            fname = f"forensic_{type_tag}_{target_name.lower()}.png"
            save_path = os.path.join(self.dirs["training"], fname)
            plt.savefig(save_path, dpi=100)
            plt.close()
            logger.info(f"💾 VisualDiagnostics: Saved {mode_str} dossier to {fname}")

        except Exception:
            logger.error(
                "VisualDiagnostics: biopsy_feature_dossier failed for '%s' — skipping plot.",
                target_name,
                exc_info=True,
            )

    def _plot_grid_with_context(
        self,
        data_5d: np.ndarray,
        feature_names: List[str],
        time_indices: np.ndarray,
        global_maps: np.ndarray,
        context_feat: str,
        offset: Tuple[int, int],
        stage_label: str,
        subdir: Optional[str] = None,
    ) -> None:
        """
        Plots the biopsy grid with a global context map sequence at the top.
        """
        n_times = data_5d.shape[0]
        n_feats = data_5d.shape[-1]

        plt, patches = _load_mpl()  # #215: lazy matplotlib (helper runs only via active methods)
        # Grid Setup: 1 row for Global Context + N rows for features
        fig = plt.figure(figsize=(4 * n_times, 3 * (n_feats + 1)))
        gs = fig.add_gridspec(n_feats + 1, n_times)

        # 1. Global Context Row (One plot per time step)
        # We need to compute vmin/vmax for the global signal
        g_vmin, g_vmax = np.nanmin(global_maps), np.nanmax(global_maps)

        # --- VISUAL PHYSICS (Fixed ADR-012 / VolumeSampler) ---
        # 1. rel_offset[0] is the relative SOUTH-most row index of the patch
        #    relative to the global volume (calculated in biopsy_sample).
        # 2. In imshow(origin='upper'), index 0 is North.
        # 3. top_rel_idx (North edge) = (GlobalHeight - 1) - (rel_offset[0] + patch_h - 1)
        # ---------------------------------------------------
        patch_h, patch_w = data_5d.shape[1], data_5d.shape[2]
        top_rel_idx = (self.height - 1) - (offset[0] + patch_h - 1)

        for t_idx in range(n_times):
            ax_g = fig.add_subplot(gs[0, t_idx])
            ax_g.imshow(
                global_maps[t_idx],
                origin="upper",
                cmap="magma",
                vmin=g_vmin,
                vmax=g_vmax,
                interpolation="nearest",
            )

            # The Box: x=relative_col, y=flipped_top_row
            rect = patches.Rectangle(
                (offset[1], top_rel_idx),
                patch_w,
                patch_h,
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
                zorder=10,
            )
            ax_g.add_patch(rect)

            ax_g.set_xticks([])
            ax_g.set_yticks([])
            if t_idx == 0:
                ax_g.set_ylabel(
                    f"GLOBAL\n{context_feat}",
                    rotation=0,
                    labelpad=60,
                    fontsize=9,
                    fontweight="bold",
                )
            ax_g.set_title(f"World T={time_indices[t_idx]}")

        # 2. Standard Biopsy Rows
        for f_idx in range(n_feats):
            feat_name = feature_names[f_idx]
            feat_slice = data_5d[..., f_idx]
            vmin, vmax = np.nanmin(feat_slice), np.nanmax(feat_slice)

            for t_idx in range(n_times):
                ax = fig.add_subplot(gs[f_idx + 1, t_idx])
                img_data = data_5d[t_idx, ..., f_idx]
                stats = self._calculate_stats(img_data)

                ax.imshow(
                    img_data,
                    origin="upper",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])

                if t_idx == 0:
                    ax.set_ylabel(f"{feat_name}", rotation=0, labelpad=60, fontsize=9)

                ax.text(
                    0.05,
                    0.95,
                    stats,
                    transform=ax.transAxes,
                    color="white",
                    fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
                )

        plt.suptitle(f"Visual Biopsy (Contextual): {stage_label}", fontsize=16)
        # rect=[left, bottom, right, top] - reserve top 5% for suptitle
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        # Sanitize label for filesystem
        safe_label = stage_label.lower().replace(" ", "_").replace("/", "_")

        # ADR 045 Routing
        target_dir = subdir if subdir else self.save_dir
        save_path = os.path.join(target_dir, f"biopsy_{safe_label}.png")

        plt.savefig(save_path, dpi=100)
        plt.close()
        logger.info(f"📸 VisualDiagnostics: Saved {save_path}")

    def _plot_grid(
        self,
        data_5d: np.ndarray,
        feature_names: List[str],
        time_indices: np.ndarray,
        stage_label: str,
        subdir: Optional[str] = None,
    ) -> None:
        """
        Core Plotting Logic.
        data_5d: [5_Times, H, W, N_Features]
        """
        n_times = data_5d.shape[0]
        n_feats = data_5d.shape[-1]

        plt, _ = _load_mpl()  # #215: lazy matplotlib (this helper only runs via active methods)
        # Plot Setup: Rows = Features, Cols = Time
        fig, axes = plt.subplots(n_feats, n_times, figsize=(4 * n_times, 3 * n_feats))
        if n_feats == 1:
            axes = np.expand_dims(axes, 0)
        if n_times == 1:
            axes = np.expand_dims(axes, 1)

        for f_idx in range(n_feats):
            feat_name = feature_names[f_idx]

            # Normalize scale per feature (across all times)
            feat_slice = data_5d[..., f_idx]
            vmin, vmax = np.nanmin(feat_slice), np.nanmax(feat_slice)

            for t_idx in range(n_times):
                ax = axes[f_idx, t_idx]
                img_data = data_5d[t_idx, ..., f_idx]

                # Stats Overlay (Hypothesis 2 Probe)
                stats = self._calculate_stats(img_data)

                # HydraNet data is already flipped to North-Up in VolumeHandler.
                # imshow(origin='upper') places index 0 (North) at the top.
                ax.imshow(
                    img_data,
                    origin="upper",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])

                if f_idx == 0:
                    ax.set_title(f"T={time_indices[t_idx]}")
                if t_idx == 0:
                    ax.set_ylabel(f"{feat_name}", rotation=0, labelpad=60, fontsize=9)

                # Add stats as text inside plot for ALL columns
                ax.text(
                    0.05,
                    0.95,
                    stats,
                    transform=ax.transAxes,
                    color="white",
                    fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
                )

        plt.suptitle(f"Visual Biopsy: {stage_label}", fontsize=16)
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        # Sanitize label for filesystem
        safe_label = stage_label.lower().replace(" ", "_").replace("/", "_")

        # ADR 045 Routing
        target_dir = subdir if subdir else self.save_dir
        save_path = os.path.join(target_dir, f"biopsy_{safe_label}.png")

        plt.savefig(save_path, dpi=100)
        plt.close()
        logger.info(f"📸 VisualDiagnostics: Saved {save_path}")

    def _calculate_stats(self, data: np.ndarray) -> str:
        """Returns μ, σ, min, max string."""
        valid = data[np.isfinite(data)]
        if valid.size == 0:
            return "EMPTY"
        mu = np.mean(valid)
        sigma = np.std(valid)
        mn = np.min(valid)
        mx = np.max(valid)
        return f"μ:{mu:.2f}\nσ:{sigma:.2f}\n[{mn:.2f}, {mx:.2f}]"
