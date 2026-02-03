"""
DataSniffer: Passive observation and strict validation for the HydraNet pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)

class DataSniffer:
    """
    A passive data observer that enforces strict contract compliance. 
    
    Identifies divergent or corrupted data states early without modifying data.
    Any contract violation results in an immediate exception (Fail Loud and Proud).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize with the configuration to know the expected state.
        """
        self.config = config
        
        # 1. Enforce Mandatory Roles from Config
        required = ["identity_cols", "time_col", "id_col", "spatial_cols", "height", "width"]
        missing = [k for k in required if k not in config]
        if missing:
            raise KeyError(f"[CRITICAL CONFIG ERROR] DataSniffer: Missing mandatory keys {missing}")
            
        self.identity_cols = config["identity_cols"]

    def sniff_ingestion(self, df: pd.DataFrame) -> None:
        """
        Suite of checks performed immediately after data is fetched from disk.
        """
        logger.info("DataSniffer: Starting Ingestion Suite (Raw Space)")
        
        self._check_obligatory_columns(df)
        self._check_spatiotemporal_uniqueness(df)
        self._check_identity_values(df)
        self._check_non_finite(df)
        
        logger.info("DataSniffer: Ingestion Suite Passed.")

    def sniff_forecast_alignment(
        self, 
        df: pd.DataFrame, 
        handler: VolumeHandler, 
        is_forecast: bool = True
    ) -> None:
        """
        Validates the temporal continuity and geographic anchoring of a volume carrier.
        """
        logger.info(f"DataSniffer: Starting {'Forecast' if is_forecast else 'History'} Alignment Suite")
        
        # Pull Ledger roles
        time_col = handler.time_col
        y_col, x_col = handler.spatial_cols
        
        max_month_df = df[time_col].max()
        min_month_df = df[time_col].min()
        
        # 1. Resolve temporal index via Ledger
        try:
            m_idx = handler.channel_map.index(time_col)
        except ValueError:
             raise ValueError(f"[CRITICAL DATA ERROR] DataSniffer: '{time_col}' missing from Handler Ledger!")

        # Pull temporal range from data
        vol_data = handler.data
        m_chan = vol_data[..., m_idx]
            
        min_month_vol = m_chan.min().item() if torch.is_tensor(m_chan) else m_chan.min()
        max_month_vol = m_chan.max().item() if torch.is_tensor(m_chan) else m_chan.max()

        # 2. Check Continuity (ADR 018 Section 1.2)
        if is_forecast:
            expected_min = max_month_df + 1
            if min_month_vol != expected_min:
                error_msg = (
                    f"[CRITICAL DATA ERROR] DataSniffer: Forecast Continuity Broken!\n"
                    f"History ends at {max_month_df}. Forecast starts at {min_month_vol} (Expected {expected_min})."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            # History Volume Check
            if min_month_vol != min_month_df or max_month_vol != max_month_df:
                error_msg = (
                    f"[CRITICAL DATA ERROR] DataSniffer: History Volume Mismatch!\n"
                    f"DF range: [{min_month_df}, {max_month_df}]\n"
                    f"Vol range: [{min_month_vol}, {max_month_vol}]"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # 3. Geographic Anchor Check (Absolute Anchoring)
        r_off, c_off = handler.spatial_offset
        if df[y_col].min() < r_off or df[x_col].min() < c_off:
             raise ValueError(
                 f"[CRITICAL DATA ERROR] DataSniffer: Geographic Anchor Violation!\n"
                 f"Data starts at ({df[y_col].min()}, {df[x_col].min()}), but "
                 f"Handler is anchored at ({r_off}, {c_off})."
             )

        logger.info("DataSniffer: Temporal and Geographic Alignment Passed.")

    def _check_identity_values(self, df: pd.DataFrame) -> None:
        """
        Orchestrates strict numerical constraints on identity columns.
        """
        time_col = self.config["time_col"]
        id_col = self.config["id_col"]
        y_col, x_col = self.config["spatial_cols"]

        self._check_finiteness(df, [time_col, id_col, y_col, x_col])
        self._check_spatial_bounds(df, y_col, x_col)

    def _check_finiteness(self, df: pd.DataFrame, cols: list[str]) -> None:
        """Enforces that essential columns are finite."""
        for col in cols:
            if not np.isfinite(df[col]).all():
                raise ValueError(f"DataSniffer: Non-finite values detected in mandatory column '{col}'")

    def _check_spatial_bounds(self, df: pd.DataFrame, y_col: str, x_col: str) -> None:
        """Verifies that the span of indices fits within the configured volume resolution."""
        height = self.config["height"]
        width = self.config["width"]
        
        r_span = df[y_col].max() - df[y_col].min()
        c_span = df[x_col].max() - df[x_col].min()
        
        if r_span >= height or c_span >= width:
            raise ValueError(
                f"[CRITICAL DATA ERROR] DataSniffer: Spatial Span Violation!\n"
                f"Data spans {r_span}x{c_span}, but volume resolution is {height}x{width}."
            )

    def _check_spatiotemporal_uniqueness(self, df: pd.DataFrame) -> None:
        """
        Enforces that each (time, id) combination is unique.
        """
        time_col = self.config["time_col"]
        id_col = self.config["id_col"]
        subset = [time_col, id_col]
        
        if df.duplicated(subset=subset).any():
            duplicates = df[df.duplicated(subset=subset, keep=False)]
            example_ids = duplicates[subset].head(5).to_dict('records')
            
            error_msg = (
                f"\n[CRITICAL DATA ERROR] DataSniffer: Duplicate Entries Detected!\n"
                f"Each combination of {subset} must be unique.\n"
                f"Found {len(duplicates)} duplicate rows. Examples: {example_ids}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _check_obligatory_columns(self, df: pd.DataFrame) -> None:
        """
        Ensures all required identity and feature columns exist.
        """
        required_cols = list(set(self.identity_cols + self.config["features"]))
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            error_msg = (
                f"\n[CRITICAL DATA ERROR] DataSniffer: Missing Obligatory Columns!\n"
                f"Missing: {missing_cols}\n"
                f"Available: {df.columns.tolist()}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _check_non_finite(self, df: pd.DataFrame) -> None:
        """
        Scans for NaNs and Infs in all numeric columns.
        """
        offending_columns = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if not np.isfinite(df[col]).all():
                    offending_columns.append(col)

        if offending_columns:
            error_details = []
            for col in offending_columns:
                n_nans = df[col].isna().sum()
                n_infs = (~np.isfinite(df[col])).sum() - n_nans
                error_details.append(f" - {col}: {n_nans} NaNs, {n_infs} Infs")
            
            error_msg = (
                "\n[CRITICAL DATA ERROR] DataSniffer detected non-finite values!\n"
                "Offending Columns:\n" + "\n".join(error_details)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)