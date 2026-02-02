"""
DataSniffer: Passive observation and strict validation for the HydraNet pipeline.
"""

import logging
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

class DataSniffer:
    """
    A passive data observer that enforces strict contract compliance. 
    
    This class identifies divergent or corrupted data states early. 
    It is strictly 'read-only' and will never modify data.
    
    Any contract violation results in an immediate exception to stop the run.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize with the configuration to know the expected state.
        """
        self.config = config
        
        # 1. Enforce Mandatory Identity Columns from Config
        if "identity_cols" not in config:
            error_msg = "[CRITICAL CONFIG ERROR] DataSniffer: 'identity_cols' missing from configuration!"
            logger.error(error_msg)
            raise KeyError(error_msg)
            
        self.identity_cols = config["identity_cols"]
        
        # 2. Minimum validation: We expect exactly 5 identities for this architecture
        if len(self.identity_cols) != 5:
            error_msg = (
                f"[CRITICAL CONFIG ERROR] DataSniffer: Identity Contract Violation!\n"
                f"Expected 5 identity columns, got {len(self.identity_cols)}: {self.identity_cols}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def sniff_ingestion(self, df: pd.DataFrame) -> None:
        """
        Suite of checks performed immediately after data is fetched from disk.
        
        Runs:
        1. Obligatory Column Check (Identity + Features)
        2. Spatiotemporal Uniqueness Check (No duplicates)
        3. Identity Value Sanity Check (Ranges and Finiteness)
        4. Non-Finite Value Check
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
        vol: Union[np.ndarray, torch.Tensor], 
        month_range: int = 36,
        is_forecast: bool = True
    ) -> None:
        """
        Validates the temporal continuity of a volume against observed history.
        
        Args:
            df: Observed historical DataFrame.
            vol: Volume carrier [T, H, W, C] (ndarray) or [B, T, C, H, W] (tensor).
            month_range: Expected length of the volume.
            is_forecast: If True, expects vol to start at history_end + 1.
                         If False, expects vol to match history exactly.
        """
        logger.info(f"DataSniffer: Starting {'Forecast' if is_forecast else 'History'} Alignment Suite")
        
        max_month_df = df["month_id"].max()
        min_month_df = df["month_id"].min()
        
        # 1. Get volume bounds from channel 3 (month_id)
        # We assume month_id is dense (via VolumeHandler)
        
        # Resolve month_id index dynamically
        channel_map = self.identity_cols + self.config.get("features", [])
        try:
            m_idx = channel_map.index("month_id")
        except ValueError:
             raise ValueError("[CRITICAL CONFIG ERROR] 'month_id' not found in channel map!")

        if torch.is_tensor(vol):
            m_chan = vol[:, :, m_idx, :, :]
        else:
            m_chan = vol[..., m_idx]
            
        min_month_vol = m_chan.min().item() if torch.is_tensor(m_chan) else m_chan.min()
        max_month_vol = m_chan.max().item() if torch.is_tensor(m_chan) else m_chan.max()

        # 2. Check Continuity
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

        logger.info("DataSniffer: Temporal Alignment Passed.")

    def _check_identity_values(self, df: pd.DataFrame) -> None:
        """
        Orchestrates strict numerical constraints on identity columns.
        """
        self._check_priogrid_gid(df)
        self._check_col(df)
        self._check_row(df)
        self._check_month_id(df)
        self._check_c_id(df)

    def _check_priogrid_gid(self, df: pd.DataFrame) -> None:
        """Enforces that priogrid_gid is positive and finite."""
        pg_col = "priogrid_gid"
        if not (df[pg_col] > 0).all() or not np.isfinite(df[pg_col]).all():
            bad_values = df[~((df[pg_col] > 0) & np.isfinite(df[pg_col]))][pg_col].unique()
            error_msg = (
                f"[CRITICAL DATA ERROR] DataSniffer: Invalid {pg_col} detected!\n"
                f"IDs must be positive and finite. Found: {bad_values[:10]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _check_month_id(self, df: pd.DataFrame) -> None:
        """Enforces that month_id is within the allowed range and finite."""
        m_col = "month_id"
        m_min, m_max = 120, 1000
        within_range = (df[m_col] >= m_min) & (df[m_col] <= m_max)
        
        if not within_range.all() or not np.isfinite(df[m_col]).all():
            bad_values = df[~(within_range & np.isfinite(df[m_col]))][m_col].unique()
            error_msg = (
                f"[CRITICAL DATA ERROR] DataSniffer: Invalid {m_col} detected!\n"
                f"Months must be between {m_min} and {m_max} and finite. Found: {bad_values[:10]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _check_col(self, df: pd.DataFrame) -> None:
        """Verifies that the span of column indices fits within the volume width."""
        width = self.config.get("width", 180)
        
        c_min, c_max = df["col"].min(), df["col"].max()
        span = c_max - c_min
        
        if span >= width:
            error_msg = (
                f"[CRITICAL DATA ERROR] DataSniffer: 'col' span too large!\n"
                f"Data spans {span} columns (min={c_min}, max={c_max}), but the "
                f"volume fixture width is {width}. Data cannot fit."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _check_row(self, df: pd.DataFrame) -> None:
        """Verifies that the span of row indices fits within the volume height."""
        height = self.config.get("height", 180)
        
        r_min, r_max = df["row"].min(), df["row"].max()
        span = r_max - r_min
        
        if span >= height:
            error_msg = (
                f"[CRITICAL DATA ERROR] DataSniffer: 'row' span too large!\n"
                f"Data spans {span} rows (min={r_min}, max={r_max}), but the "
                f"volume fixture height is {height}. Data cannot fit."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _check_c_id(self, df: pd.DataFrame) -> None:
        """Placeholder for c_id sanity check."""
        pass

    def _check_spatiotemporal_uniqueness(self, df: pd.DataFrame) -> None:
        """
        Enforces that each (month_id, priogrid_gid) combination is unique.
        Duplicate entries would corrupt the 4D volume transformation.
        """
        subset = ["month_id", "priogrid_gid"]
        if df.duplicated(subset=subset).any():
            duplicates = df[df.duplicated(subset=subset, keep=False)]
            # Format some examples for the log
            example_ids = duplicates[subset].head(5).to_dict('records')
            
            error_msg = (
                f"\n[CRITICAL DATA ERROR] DataSniffer: Duplicate Entries Detected!\n"
                f"Each combination of {subset} must be unique.\n"
                f"Found {len(duplicates)} duplicate rows. Examples: {example_ids}\n"
                f"Check your queryset for overlapping data sources."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _check_obligatory_columns(self, df: pd.DataFrame) -> None:
        """
        Ensures all required identity and feature columns exist.
        """
        # 1. Identity Columns (from config)
        identity_cols = self.identity_cols
        
        # 2. Feature Columns (from config)
        if "features" not in self.config:
            error_msg = "[CRITICAL CONFIG ERROR] DataSniffer: 'features' missing from configuration!"
            logger.error(error_msg)
            raise KeyError(error_msg)
            
        feature_cols = self.config["features"]

        required_cols = list(set(identity_cols + feature_cols))
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
        Scans for NaNs and Infs.
        """
        offending_columns = []
        for col in df.columns:
            # We only check numeric columns for finiteness
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
                f"\n[CRITICAL DATA ERROR] DataSniffer detected non-finite values!\n"
                f"Offending Columns:\n" + "\n".join(error_details)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
