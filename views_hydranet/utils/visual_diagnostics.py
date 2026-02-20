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
            
            # Select Features (Metadata + Features)
            interesting = []
            
            # 1. Metadata / Identity Group
            # We want: month_id, priogrid_gid, c_id, row, col
            meta_order = [vh.time_col, vh.id_col, "c_id"] + list(vh.spatial_cols)
            for c in meta_order:
                if c in vh.channel_map and c not in interesting:
                    interesting.append(c)
            
            # 2. Features Group (All linear signals)
            # Pull directly from ledger to ensure completeness (ADR 012)
            for c in vh._metadata.feature_cols:
                if c in vh.channel_map and c not in interesting:
                    interesting.append(c)
            
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
            # 1. Select interesting features (Same as biopsy_volume)
            interesting = []
            meta_order = [sample_vh.time_col, sample_vh.id_col, "c_id"] + list(sample_vh.spatial_cols)
            for c in meta_order:
                if c in sample_vh.channel_map: interesting.append(c)
            for c in sample_vh._metadata.feature_cols:
                if c in sample_vh.channel_map and c not in interesting:
                    interesting.append(c)

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

            # 3. Extract Global Context for the FIRST interesting feature (usually month_id or conflict)
            # We use the middle time step of the sample for the global view
            mid_t = t_indices[2]
            g_data = global_vh.data
            g_t_idx = global_vh.get_axis_idx("T")
            g_c_idx = global_vh.get_axis_idx("C")
            
            # Identify first signal feature
            signal_feat = interesting[-1] 
            g_feat_idx = global_vh.channel_map.index(signal_feat)
            id_feat_idx = global_vh.channel_map.index(global_vh.id_col)
            
            global_map = g_data[mid_t, ..., g_feat_idx] if g_t_idx == 0 else g_data[:, :, mid_t, g_feat_idx]
            # Ensure correct orientation for plotting
            if g_t_idx == 0: # [T, H, W, C]
                 global_map = g_data[mid_t, ..., g_feat_idx]
                 id_map = g_data[mid_t, ..., id_feat_idx]
            else: # [H, W, T, C] or other
                 # Fallback to general slice
                 slc = [slice(None)] * g_data.ndim
                 slc[g_t_idx] = mid_t
                 slc[g_c_idx] = g_feat_idx
                 global_map = g_data[tuple(slc)]
                 slc[g_c_idx] = id_feat_idx
                 id_map = g_data[tuple(slc)]

            # Mask Ocean (priogrid_gid <= 0)
            global_map = np.where(id_map > 0, global_map, np.nan)

            # 4. Plot with Context
            self._plot_grid_with_context(
                sample_slice, 
                interesting, 
                t_indices, 
                global_map, 
                signal_feat,
                sample_vh.spatial_offset, 
                stage_label
            )

        except Exception as e:
            logger.error(f"VisualDiagnostics: Failed to biopsy sample at {stage_label}: {e}")

    def _plot_grid_with_context(self, data_5d, feature_names, time_indices, global_map, context_feat, offset, stage_label):
        """
        Plots the biopsy grid with a global context map at the top.
        """
        import matplotlib.patches as patches
        n_times = data_5d.shape[0]
        n_feats = data_5d.shape[-1]
        
        # Grid Setup: 1 row for Global Context + N rows for features
        fig = plt.figure(figsize=(4 * n_times, 3 * (n_feats + 2)))
        gs = fig.add_gridspec(n_feats + 2, n_times)

        # 1. Global Context Row (Spans all columns)
        ax_global = fig.add_subplot(gs[0:2, :])
        im = ax_global.imshow(global_map, origin='upper', cmap='magma', interpolation='nearest')
        ax_global.set_title(f"Global Context: {context_feat} (Orange Box = Sample Location)")
        
        # Draw Orange Bounding Box
        # VolumeHandler flip(axis=0) means North is at index 0. 
        # But offsets are raw. 
        # r_idx = row - row_offset. 
        # In North-Up array, top is max-latitude.
        # This part is tricky. We'll eyeball it.
        patch_h, patch_w = data_5d.shape[1], data_5d.shape[2]
        # Calculate top-left corner in array coordinates
        # Offset is (row_off, col_off). 
        # If global height is 180, and row_off is 100, then array[0] is latitude 180+100? No.
        # Let's trust the relative positioning.
        rect = patches.Rectangle((offset[1], offset[0]), patch_w, patch_h, linewidth=2, edgecolor='orange', facecolor='none')
        ax_global.add_patch(rect)
        plt.colorbar(im, ax=ax_global)

        # 2. Standard Biopsy Rows
        for f_idx in range(n_feats):
            feat_name = feature_names[f_idx]
            feat_slice = data_5d[..., f_idx]
            vmin, vmax = np.nanmin(feat_slice), np.nanmax(feat_slice)
            
            for t_idx in range(n_times):
                ax = fig.add_subplot(gs[f_idx + 2, t_idx])
                img_data = data_5d[t_idx, ..., f_idx]
                stats = self._calculate_stats(img_data)
                
                ax.imshow(img_data, origin='upper', cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                
                if f_idx == 0:
                    ax.set_title(f"T={time_indices[t_idx]}")
                if t_idx == 0:
                    ax.set_ylabel(f"{feat_name}", rotation=0, labelpad=60, fontsize=9)
                
                ax.text(0.05, 0.95, stats, transform=ax.transAxes, color='white', 
                        fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        plt.suptitle(f"Visual Biopsy (Contextual): {stage_label}", fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f"biopsy_{stage_label.lower().replace(' ', '_')}.png")
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
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f"biopsy_{stage_label.lower().replace(' ', '_')}.png")
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
