"""
VisualDiagnostics: The Visual Truth Engine.
Governed by the Hybrid Diagnostics Plan (2026-02-19).
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)

class VisualDiagnostics:
    """
    A unified observer for generating 'Visual Biopsies' of spatiotemporal data
    at every stage of the pipeline.
    
    Implements the 'Null Object' pattern if diagnostic_visualizations is False.
    """

    def __init__(self, config: Dict[str, Any], run_timestamp: Optional[str] = None) -> None:
        self.active = config.get("diagnostic_visualizations", False)
        self.config = config # Store for dynamic labelling
        
        if self.active:
            # Determine save directory
            # If manager passed a timestamp in config (ADR 026), use it.
            from datetime import datetime
            config_ts = config.get("model_time_stamp")
            ts = run_timestamp or config_ts or datetime.now().strftime("%d%m%y_%H%M")
            self.save_dir = f"reports/plots/diagnostics/{ts}"
            os.makedirs(self.save_dir, exist_ok=True)
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

    def biopsy_dataframe(self, df: pd.DataFrame, stage_label: str, features: List[str] = []) -> None:
        """
        Adapts a flat or MultiIndexed DataFrame to the biopsy grid.
        """
        if not self.active: return

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
            
            # We need to construct a mini-volume for plotting
            vol_slice = np.zeros((5, self.height, self.width, len(features)))
            vol_slice[:] = np.nan
            
            for i, feat in enumerate(features):
                if feat not in df_slice.columns:
                    # Check if it was part of the index and is now a column
                    continue
                    
                r_idx = (df_slice[self.y_col] - self.row_offset).astype(int).values
                c_idx = (df_slice[self.x_col] - self.col_offset).astype(int).values
                
                month_map = {m: idx for idx, m in enumerate(global_months)}
                t_idx = df_slice[self.time_col].map(month_map).values
                
                vol_slice[t_idx, r_idx, c_idx, i] = df_slice[feat].values

            # 3. Orient North-Up (Match VolumeHandler logic)
            # Fancy indexing places min-latitude at index 0. 
            # We flip axis 1 (Height) so North is at index 0 for 'upper' plotting.
            vol_slice = np.flip(vol_slice, axis=1)

            # Plot
            self._plot_grid(vol_slice, features, global_months, stage_label)

        except Exception as e:
            logger.error(f"VisualDiagnostics: Failed to biopsy DataFrame at {stage_label}: {e}")
            # Raise in debug mode to see stack trace, or log clearly
            if logger.getEffectiveLevel() <= logging.DEBUG:
                raise e

    def biopsy_volume(self, vh: VolumeHandler, stage_label: str) -> None:
        """
        Adapts a VolumeHandler to the biopsy grid.
        This is the most trusted view because it reflects the internal array state.
        """
        if not self.active: return

        try:
            # 1. Extract Data [T, H, W, C]
            data = vh.data.detach().cpu().numpy() if torch.is_tensor(vh.data) else vh.data
            
            # Handle Stochastic [T, H, W, C, S] -> Mean -> [T, H, W, C]
            if "S" in vh.axes:
                s_idx = vh.get_axis_idx("S")
                data = np.nanmean(data, axis=s_idx)
            
            # Ensure T, H, W, C order
            t_idx, h_idx, w_idx, c_idx = (
                vh.get_axis_idx("T"), vh.get_axis_idx("H"), vh.get_axis_idx("W"), vh.get_axis_idx("C")
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
            
            self._plot_grid(vol_slice, interesting, t_indices, stage_label)

        except Exception as e:
            logger.error(f"VisualDiagnostics: Failed to biopsy Volume at {stage_label}: {e}")

    def biopsy_tensor(self, tensor: torch.Tensor, stage_label: str, channel_names: List[str]) -> None:
        """
        Adapts a PyTorch Tensor [B, T, C, H, W] to the biopsy grid.
        Assumes Batch=0.
        """
        if not self.active: return

        try:
            # [B, T, C, H, W] -> [T, H, W, C]
            if tensor.ndim == 5:
                data = tensor[0].permute(0, 2, 3, 1).detach().cpu().numpy()
            elif tensor.ndim == 4: # [T, C, H, W]
                data = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
            else:
                logger.warning(f"VisualDiagnostics: Tensor shape {tensor.shape} not supported.")
                return

            t_len = data.shape[0]
            t_indices = np.linspace(0, t_len - 1, 5, dtype=int)
            
            # Slice time
            data = data[t_indices] # [5, H, W, C]
            
            self._plot_grid(data, channel_names, t_indices, stage_label)

        except Exception as e:
            logger.error(f"VisualDiagnostics: Failed to biopsy Tensor at {stage_label}: {e}")

    def biopsy_sample(self, sample_vh: VolumeHandler, global_vh: VolumeHandler, stage_label: str) -> None:
        """
        Specialized biopsy for training samples. 
        Shows the local patch alongside its global geographic context.
        """
        if not self.active: return

        try:
            # 1. Select interesting features (Metadata first, then signals)
            interesting, meta_cols = self._select_display_channels(sample_vh)

            # 2. Extract Sample Data [5, H_patch, W_patch, F]
            s_data = sample_vh.data
            t_idx, h_idx, w_idx, c_idx = (
                sample_vh.get_axis_idx("T"), sample_vh.get_axis_idx("H"), 
                sample_vh.get_axis_idx("W"), sample_vh.get_axis_idx("C")
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
            
            signal_feat = next(
                (c for c in interesting if c not in meta_cols),
                None
            )
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
                sample_vh.spatial_offset[1] - global_vh.spatial_offset[1]
            )

            # 5. Plot with Context
            self._plot_grid_with_context(
                sample_slice, 
                interesting, 
                t_indices, 
                global_maps, 
                signal_feat,
                rel_offset, 
                stage_label
            )

        except Exception as e:
            logger.error(f"VisualDiagnostics: Failed to biopsy sample at {stage_label}: {e}")

    def _select_display_channels(self, vh: VolumeHandler) -> Tuple[List[str], set]:
        """
        Builds the ordered channel list for biopsy display.

        Returns (interesting, meta_cols) where:
        - interesting: ordered list of channel names to display (metadata first, then signals)
        - meta_cols: frozenset of identity/metadata channel names, enabling O(1)
          signal-vs-metadata classification without re-scanning the list.
        """
        meta_cols = {vh.time_col, vh.id_col, "c_id"} | set(vh.spatial_cols)
        interesting: List[str] = []

        # 1. Metadata / Identity Group (fixed display order)
        meta_order = [vh.time_col, vh.id_col, "c_id"] + list(vh.spatial_cols)
        for c in meta_order:
            if c in vh.channel_map and c not in interesting:
                interesting.append(c)

        # 2. Signal Features Group (pulled from ledger — ADR 012)
        for c in vh._metadata.feature_cols:
            if c in vh.channel_map and c not in interesting:
                interesting.append(c)

        return interesting, meta_cols

    def biopsy_autoregressive(self, truth_seq: List[np.ndarray], pred_seq: List[np.ndarray], 
                              stage_label: str, channel_names: List[str], time_indices: List[float] = None) -> None:
        """
        Specialized biopsy for the autoregressive feedback loop.
        Structure: 3 Rows (Truth, Pred, Delta) x 6 Columns (Seed + 5 steps).
        """
        if not self.active: return

        try:
            # Inputs are lists of arrays [H, W, C]
            # Convert to [Time, Row, Col, Chan]
            truth = np.stack(truth_seq) # [6, H, W, C]
            pred = np.stack(pred_seq)   # [6, H, W, C]
            delta = np.abs(truth - pred)
            
            n_times = 6
            
            # To keep it "Joyful" and not too huge, we'll plot the first signal only
            feat_idx = 0
            feat_name = channel_names[feat_idx]

            fig, axes = plt.subplots(3, n_times, figsize=(18, 10))
            
            row_labels = ["GROUND TRUTH (y)", "PREDICTION (ŷ)", "ABSOLUTE DELTA (|y-ŷ|)"]
            data_rows = [truth, pred, delta]
            
            # Shared scale for Truth and Pred, Delta gets its own
            v_min = np.min([np.nanmin(truth[..., feat_idx]), np.nanmin(pred[..., feat_idx])])
            v_max = np.max([np.nanmax(truth[..., feat_idx]), np.nanmax(pred[..., feat_idx])])
            d_max = np.nanmax(delta[..., feat_idx])

            for r_idx in range(3):
                row_data = data_rows[r_idx]
                for t_idx in range(n_times):
                    ax = axes[r_idx, t_idx]
                    img = row_data[t_idx, ..., feat_idx]
                    
                    # Style
                    cmap = 'magma' if r_idx < 2 else 'Reds'
                    vmx = v_max if r_idx < 2 else d_max
                    vmn = v_min if r_idx < 2 else 0
                    
                    ax.imshow(img, origin='upper', cmap=cmap, vmin=vmn, vmax=vmx, interpolation='nearest')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # Labels
                    if r_idx == 0:
                        m_id = int(time_indices[t_idx]) if time_indices else t_idx
                        suffix = "_seed" if t_idx == 0 else "_out"
                        ax.set_title(f"{m_id}{suffix}", fontweight='bold')
                    
                    if t_idx == 0:
                        ax.set_ylabel(row_labels[r_idx], rotation=0, labelpad=80, fontweight='bold')

            plt.suptitle(f"Autoregressive Forensic: {stage_label} ({feat_name})", fontsize=18, y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Add vertical delimitation line correctly (between col 0 and 1)
            # IMPORTANT: tight_layout must be called BEFORE get_position()
            fig.canvas.draw() 
            pos0 = axes[0, 0].get_position()
            pos1 = axes[0, 1].get_position()
            line_x = (pos0.x1 + pos1.x0) / 2
            fig.add_artist(plt.Line2D([line_x, line_x], [0.05, 0.9], color='cyan', linestyle='--', linewidth=2, alpha=0.8))
            
            # Sanitize label for filesystem
            safe_label = stage_label.lower().replace(' ', '_').replace('/', '_')
            save_path = os.path.join(self.save_dir, f"biopsy_{safe_label}.png")
            plt.savefig(save_path, dpi=100)
            plt.close()
            logger.info(f"📸 VisualDiagnostics: Saved {save_path}")

        except Exception as e:
            logger.error(f"VisualDiagnostics: Failed to biopsy autoregressive loop at {stage_label}: {e}")

    def biopsy_training_performance(self, 
                                   y_reg: np.ndarray, y_hat_reg: np.ndarray, 
                                   y_cls: np.ndarray, y_hat_cls: np.ndarray, 
                                   stage_label: str,
                                   time_indices: List[float] = None) -> None:
        """
        4x6 Forensic Grid for Training runs.
        Rows: [Y_Reg, Y_Hat_Reg, Y_Cls, Y_Hat_Cls]
        Cols: 6 sequential time steps.
        """
        if not self.active: return

        try:
            # Inputs are [T, H, W, C]
            n_times = min(6, y_reg.shape[0])
            
            fig, axes = plt.subplots(4, n_times, figsize=(18, 12))
            
            row_labels = [
                "GROUND TRUTH (Reg)", "PREDICTION (Reg)",
                "GROUND TRUTH (Cls)", "PREDICTION (Cls)"
            ]
            data_rows = [y_reg, y_hat_reg, y_cls, y_hat_cls]
            cmaps = ['magma', 'magma', 'viridis', 'viridis']
            
            for r_idx in range(4):
                row_data = data_rows[r_idx]
                feat_slice = row_data[..., 0] # Use first target only
                vmin, vmax = np.nanmin(feat_slice), np.nanmax(feat_slice)
                
                for t_idx in range(n_times):
                    ax = axes[r_idx, t_idx]
                    img = feat_slice[t_idx]
                    stats = self._calculate_stats(img)
                    
                    ax.imshow(img, origin='upper', cmap=cmaps[r_idx], vmin=vmin, vmax=vmax, interpolation='nearest')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    if r_idx == 0:
                        m_id = int(time_indices[t_idx]) if time_indices else t_idx
                        # Midpoint split: 3 in, 3 out
                        suffix = "_in" if t_idx < 3 else "_out"
                        ax.set_title(f"{m_id}{suffix}", fontweight='bold')
                    if t_idx == 0:
                        ax.set_ylabel(f"{row_labels[r_idx]}\n{stats}", rotation=0, labelpad=80, fontsize=9, fontweight='bold')
                    else:
                        ax.text(0.05, 0.95, stats, transform=ax.transAxes, color='white', 
                                fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

            plt.suptitle(f"Training Performance Forensic: {stage_label}", fontsize=18, y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Vertical line between History (3) and Forecast (3)
            if n_times > 3:
                fig.canvas.draw()
                pos2 = axes[0, 2].get_position()
                pos3 = axes[0, 3].get_position()
                line_x = (pos2.x1 + pos3.x0) / 2
                fig.add_artist(plt.Line2D([line_x, line_x], [0.05, 0.9], color='cyan', linestyle='--', linewidth=2, alpha=0.8))
            
            safe_label = stage_label.lower().replace(' ', '_').replace('/', '_')
            save_path = os.path.join(self.save_dir, f"biopsy_{safe_label}.png")
            plt.savefig(save_path, dpi=100)
            plt.close()
            logger.info(f"📸 VisualDiagnostics: Saved {save_path}")

        except Exception as e:
            logger.error(f"VisualDiagnostics: Failed training biopsy at {stage_label}: {e}")

    def biopsy_loss_curves(self, 
                           history_reg: List[float], 
                           history_cls: List[float], 
                           history_total: List[float], 
                           stage_label: str) -> None:
        """
        Generates both Linear and Log-scale plots showing loss evolution over lessons.
        """
        if not self.active: return

        def _generate_plot(is_log: bool):
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            
            l_reg = self.config.get("loss_reg", "Unknown")
            l_cls = self.config.get("loss_class", "Unknown")
            
            titles = [f"Regression Loss ({l_reg})", f"Classification Loss ({l_cls})", "Total Multi-Task Loss"]
            data = [history_reg, history_cls, history_total]
            colors = ['firebrick', 'seagreen', 'royalblue']
            
            for i in range(3):
                ax = axes[i]
                # Filter out non-positive for log plot to avoid matplotlib warnings
                plot_data = data[i]
                ax.plot(plot_data, marker='o', linestyle='-', color=colors[i], markersize=4, alpha=0.7)
                
                scale_suffix = " (Log Scale)" if is_log else ""
                ax.set_title(titles[i] + scale_suffix, fontweight='bold')
                ax.set_xlabel("Lesson")
                ax.set_ylabel("Loss")
                ax.grid(True, which="both", alpha=0.3)
                
                if is_log:
                    ax.set_yscale('log')
                
                if len(plot_data) > 0:
                    current = plot_data[-1]
                    ax.text(0.95, 0.95, f"Current: {current:.4f}", transform=ax.transAxes, 
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.suptitle(f"Learning Dynamics: {stage_label}", fontsize=18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            fname = "loss_evolution_log.png" if is_log else "loss_evolution.png"
            save_path = os.path.join(self.save_dir, fname)
            plt.savefig(save_path, dpi=100)
            plt.close()

        try:
            _generate_plot(is_log=False)
            _generate_plot(is_log=True)
        except Exception as e:
            logger.error(f"VisualDiagnostics: Failed to plot loss curves: {e}")

    def biopsy_feature_dossier(self, target_name: str, dossier: Dict[str, List[float]], 
                               stage_label: str, target_type: str = "REG") -> None:
        """
        Generates Joyful Forensic Dossiers.
        One-metric-per-row layout.
        """
        if not self.active: return

        is_reg = (target_type == "REG")
        metrics = self.config.get("regression_metrics", []) if is_reg else self.config.get("classification_metrics", [])
        
        # Filter for metrics actually in the dossier
        active_metrics = [m for m in metrics if m in dossier]
        
        # Determine Rows: N metrics + Magnitude + Bias
        n_metrics = len(active_metrics)
        n_extra = 2 if is_reg else 1 # Reg gets Mag+Bias, Cls gets Bias
        total_rows = n_metrics + n_extra
        
        try:
            fig, axes = plt.subplots(total_rows, 1, figsize=(10, 4 * total_rows))
            if total_rows == 1: axes = [axes]

            # 1. Plot individual metrics (One per row)
            for i, m in enumerate(active_metrics):
                ax = axes[i]
                ax.plot(dossier[m], label=m.upper(), marker='o', color='royalblue', alpha=0.7)
                ax.set_title(f"Metric: {m.upper()}", fontweight='bold')
                ax.set_ylabel("Score")
                ax.grid(True, alpha=0.3)
                if not is_reg: ax.set_ylim(0, 1.05) # Cls normalization

            # 2. Plot extra Rows
            if is_reg:
                # Row N: Magnitudes
                ax_mag = axes[n_metrics]
                ax_mag.plot(dossier["y_bar"], label="Actual Mean (y_bar)", color='black', linewidth=2)
                ax_mag.plot(dossier["y_hat_bar"], label="Pred Mean (ŷ_bar)", color='orange', linestyle='--', alpha=0.8)
                ax_mag.set_title("Magnitude Pulse (Average Counts)", fontweight='bold')
                ax_mag.legend()
                ax_mag.grid(True, alpha=0.3)

                # Row N+1: Bias
                ax_bias = axes[n_metrics + 1]
                ax_bias.plot(dossier["bias_instant"], label="Instant (Lesson)", color='firebrick', alpha=0.6)
                ax_bias.plot(dossier["bias_running"], label="Running (Global)", color='royalblue', linewidth=2)
                ax_bias.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
                ax_bias.set_title("Calibration Pulse (ŷ_bar / y_bar)", fontweight='bold')
                ax_bias.set_ylabel("Ratio")
                if any(v > 10 for v in dossier["bias_instant"]): ax_bias.set_yscale('log')
                ax_bias.legend()
                ax_bias.grid(True, alpha=0.3)
            else:
                # Cls Bias
                ax_bias = axes[n_metrics]
                ax_bias.plot(dossier["bias_instant"], label="Event Ratio (ŷ_events / y_events)", color='seagreen', alpha=0.8)
                ax_bias.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
                ax_bias.set_title("Detection Bias Pulse", fontweight='bold')
                ax_bias.legend()
                ax_bias.grid(True, alpha=0.3)

            mode_str = "REGRESSION" if is_reg else "CLASSIFICATION"
            plt.suptitle(f"{mode_str} FORENSIC: {target_name} ({stage_label})", fontsize=18)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            
            type_tag = "reg" if is_reg else "cls"
            fname = f"forensic_{type_tag}_{target_name.lower()}.png"
            save_path = os.path.join(self.save_dir, fname)
            plt.savefig(save_path, dpi=100)
            plt.close()
            logger.info(f"💾 VisualDiagnostics: Saved {mode_str} dossier to {fname}")

        except Exception as e:
            logger.error(f"VisualDiagnostics: Failed to generate joyful dossier for {target_name}: {e}")

    def _plot_grid_with_context(self, data_5d, feature_names, time_indices, global_maps, context_feat, offset, stage_label):
        """
        Plots the biopsy grid with a global context map sequence at the top.
        """
        import matplotlib.patches as patches
        n_times = data_5d.shape[0]
        n_feats = data_5d.shape[-1]
        
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
            ax_g.imshow(global_maps[t_idx], origin='upper', cmap='magma', vmin=g_vmin, vmax=g_vmax, interpolation='nearest')
            
            # The Box: x=relative_col, y=flipped_top_row
            rect = patches.Rectangle(
                (offset[1], top_rel_idx), 
                patch_w, patch_h, 
                linewidth=2, 
                edgecolor='cyan', 
                facecolor='none',
                zorder=10
            )
            ax_g.add_patch(rect)
            
            ax_g.set_xticks([])
            ax_g.set_yticks([])
            if t_idx == 0:
                ax_g.set_ylabel(f"GLOBAL\n{context_feat}", rotation=0, labelpad=60, fontsize=9, fontweight='bold')
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
                
                ax.imshow(img_data, origin='upper', cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                
                if t_idx == 0:
                    ax.set_ylabel(f"{feat_name}", rotation=0, labelpad=60, fontsize=9)
                
                ax.text(0.05, 0.95, stats, transform=ax.transAxes, color='white', 
                        fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        plt.suptitle(f"Visual Biopsy (Contextual): {stage_label}", fontsize=16)
        # rect=[left, bottom, right, top] - reserve top 5% for suptitle
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Sanitize label for filesystem
        safe_label = stage_label.lower().replace(' ', '_').replace('/', '_')
        save_path = os.path.join(self.save_dir, f"biopsy_{safe_label}.png")
        plt.savefig(save_path, dpi=100)
        plt.close()
        logger.info(f"📸 VisualDiagnostics: Saved {save_path}")

    def _plot_grid(self, data_5d: np.ndarray, feature_names: List[str], time_indices: np.ndarray, stage_label: str) -> None:
        """
        Core Plotting Logic.
        data_5d: [5_Times, H, W, N_Features]
        """
        n_times = data_5d.shape[0]
        n_feats = data_5d.shape[-1]
        
        # Plot Setup: Rows = Features, Cols = Time
        fig, axes = plt.subplots(n_feats, n_times, figsize=(4 * n_times, 3 * n_feats))
        if n_feats == 1: axes = np.expand_dims(axes, 0)
        if n_times == 1: axes = np.expand_dims(axes, 1)

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
                im = ax.imshow(img_data, origin='upper', cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                
                if f_idx == 0:
                    ax.set_title(f"T={time_indices[t_idx]}")
                if t_idx == 0:
                    ax.set_ylabel(f"{feat_name}", rotation=0, labelpad=60, fontsize=9)
                
                # Add stats as text inside plot for ALL columns
                ax.text(0.05, 0.95, stats, transform=ax.transAxes, color='white', 
                        fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        plt.suptitle(f"Visual Biopsy: {stage_label}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Sanitize label for filesystem
        safe_label = stage_label.lower().replace(' ', '_').replace('/', '_')
        save_path = os.path.join(self.save_dir, f"biopsy_{safe_label}.png")
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
