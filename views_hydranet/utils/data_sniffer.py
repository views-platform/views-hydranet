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
        Initialize with the handshake configuration to know the expected state.
        """
        self.config = config
        # Central Authority for Identity Columns
        self.identity_cols = ["priogrid_gid", "col", "row", "month_id", "c_id"]

    def sniff_ingestion(self, df: pd.DataFrame) -> None:
        """
        Suite of checks performed immediately after data is fetched from disk.
        
        Runs:
        1. Obligatory Column Check
        2. Spatiotemporal Uniqueness Check (No duplicates)
        3. Prefix Integrity Check (lr_ prefix enforcement)
        4. Identity Value Sanity Check (Ranges and Finiteness)
        5. Non-Finite Value Check
        """
        logger.info("DataSniffer: Starting Ingestion Suite (Raw Space)")
        
        self._check_obligatory_columns(df)
        self._check_spatiotemporal_uniqueness(df)
        self._check_column_prefixes(df)
        self._check_identity_values(df)
        self._check_non_finite(df)
        
        logger.info("DataSniffer: Ingestion Suite Passed.")

    def sniff_forecast_alignment(
        self, df: pd.DataFrame, forecast_storage_vol: Union[np.ndarray, torch.Tensor], month_range: int = 36
    ) -> None:
        """
        Validates that the forecast horizon is perfectly aligned with historical data.
        
        Args:
            df: Observed historical DataFrame.
            forecast_storage_vol: Forecast metadata volume [B, T, C, H, W] or [T, H, W, C].
            month_range: Expected length of the forecast.
            
        Raises:
            ValueError: If the forecast does not start exactly at last_month + 1.
        """
        logger.info("DataSniffer: Starting Forecast Alignment Suite")
        
        max_month_id_df = df["month_id"].max()
        
        # Handle both Tensor [B,T,C,H,W] and ndarray [T,H,W,C]
        if torch.is_tensor(forecast_storage_vol):
            # Tensor: Channel is at index 3 (B, T, C, H, W)
            # Channel 3 is month_id
            min_month_id_vol = forecast_storage_vol[:, :, 3, :, :].min().item()
            max_month_id_vol = forecast_storage_vol[:, :, 3, :, :].max().item()
        else:
            # ndarray: Channel is at index 3 (T, H, W, C)
            min_month_id_vol = forecast_storage_vol[:, :, :, 3].min()
            max_month_id_vol = forecast_storage_vol[:, :, :, 3].max()

        if min_month_id_vol != max_month_id_df + 1:
            error_msg = (
                f"[CRITICAL DATA ERROR] DataSniffer: Forecast Continuity Broken!\n"
                f"History ends at month {max_month_id_df}.\n"
                f"Forecast starts at month {min_month_id_vol} (Expected {max_month_id_df + 1})."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if max_month_id_vol != max_month_id_df + month_range:
            error_msg = (
                f"[CRITICAL DATA ERROR] DataSniffer: Forecast Horizon Mismatch!\n"
                f"Expected horizon: {month_range} months.\n"
                f"Actual horizon:   {max_month_id_vol - max_month_id_df} months."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("DataSniffer: Forecast Alignment Passed.")

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
        """Placeholder for col sanity check."""
        pass

    def _check_row(self, df: pd.DataFrame) -> None:
        """Placeholder for row sanity check."""
        pass

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

    def _check_column_prefixes(self, df: pd.DataFrame) -> None:
        """
        Enforces that all non-identity columns start with the 'lr_' prefix.
        """
        # Identify columns that are neither identity columns nor prefixed correctly
        offending_columns = [
            col for col in df.columns 
            if col not in self.identity_cols and not col.startswith("lr_")
        ]

        if offending_columns:
            error_msg = (
                f"\n[CRITICAL DATA ERROR] DataSniffer: Prefix Violation!\n"
                f"The following columns do not have the mandatory 'lr_' prefix: {offending_columns}\n"
                f"All non-identity columns must be explicitly labeled as raw ('lr_').\n"
                f"Auto-fixing is strictly forbidden."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _check_obligatory_columns(self, df: pd.DataFrame) -> None:
        """
        Ensures all required identity and feature columns exist.
        """
        # 1. Identity Columns
        identity_cols = self.identity_cols
        
        # 2. Feature Columns (Declarative Config)
        feature_cols = []
        for method in ["log1p", "asinh", "identity"]:
            feature_cols.extend(self.config.get(method, []))

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