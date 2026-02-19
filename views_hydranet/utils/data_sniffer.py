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
            err_msg = f"[CRITICAL CONFIG ERROR] DataSniffer: Missing mandatory keys {missing}"
            
            logger.error(err_msg)
            
            raise KeyError(err_msg)

        self.identity_cols = config["identity_cols"]

    def sniff_ingestion(self, df: pd.DataFrame) -> None:
        """
        Suite of checks performed immediately after data is fetched from disk.
        """
        print("")
        logger.info("DataSniffer: Starting Ingestion Suite (Raw Space)")

        self._check_obligatory_columns(df)
        self._check_spatiotemporal_uniqueness(df)
        self._check_identity_values(df)
        self._check_non_finite(df)

        logger.info("DataSniffer: Ingestion Suite Passed.")
        print("")

    def sniff_forecast_alignment(
        self,
        df: pd.DataFrame,
        handler: VolumeHandler,
        is_forecast: bool = True
    ) -> None:
        """
        Validates the temporal continuity and geographic anchoring of a volume carrier.
        """
        print("")
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
             err_msg = f"[CRITICAL DATA ERROR] DataSniffer: '{time_col}' missing from Handler Ledger!"
             
             logger.error(err_msg)
             
             raise ValueError(err_msg)

        # Pull temporal range from data
        vol_data = handler.data
        m_chan = vol_data[..., m_idx]

        min_month_vol = m_chan.min().item() if torch.is_tensor(m_chan) else m_chan.min()
        max_month_vol = m_chan.max().item() if torch.is_tensor(m_chan) else m_chan.max()

        # 2. Check Continuity (ADR 018 Section 1.2)
        if is_forecast:
            expected_min = max_month_df + 1
            if min_month_vol != expected_min:
                err_msg = (
                    f"[CRITICAL DATA ERROR] DataSniffer: Forecast Continuity Broken!\n"
                    f"History ends at {max_month_df}. Forecast starts at {min_month_vol} (Expected {expected_min})."
                )
                
                logger.error(err_msg)
                
                raise ValueError(err_msg)
        else:
            # History Volume Check
            if min_month_vol != min_month_df or max_month_vol != max_month_df:
                err_msg = (
                    f"[CRITICAL DATA ERROR] DataSniffer: History Volume Mismatch!\n"
                    f"DF range: [{min_month_df}, {max_month_df}]\n"
                    f"Vol range: [{min_month_vol}, {max_month_vol}]"
                )
                
                logger.error(err_msg)
                
                raise ValueError(err_msg)

        # 3. Geographic Anchor Check (Absolute Anchoring)
        r_off, c_off = handler.spatial_offset
        if df[y_col].min() < r_off or df[x_col].min() < c_off: 
            err_msg = (
                f"[CRITICAL DATA ERROR] DataSniffer: Geographic Anchor Violation!\n"
                f"Data starts at ({df[y_col].min()}, {df[x_col].min()}), but "
                f"Handler is anchored at ({r_off}, {c_off})."
            )
            
            logger.error(err_msg)
            
            raise ValueError(err_msg)

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
                err_msg = f"DataSniffer: Non-finite values detected in mandatory column '{col}'"
                
                logger.error(err_msg)
                
                raise ValueError(err_msg)

    def _check_spatial_bounds(self, df: pd.DataFrame, y_col: str, x_col: str) -> None:
        """Verifies that the span of indices fits within the configured volume resolution."""
        height = self.config["height"]
        width = self.config["width"]

        r_span = df[y_col].max() - df[y_col].min()
        c_span = df[x_col].max() - df[x_col].min()

        if r_span >= height or c_span >= width:
            err_msg = (
                f"[CRITICAL DATA ERROR] DataSniffer: Spatial Span Violation!\n"
                f"Data spans {r_span}x{c_span}, but volume resolution is {height}x{width}."
            )
            
            logger.error(err_msg)
            
            raise ValueError(err_msg)

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

            err_msg = (
                f"\n[CRITICAL DATA ERROR] DataSniffer: Duplicate Entries Detected!\n"
                f"Each combination of {subset} must be unique.\n"
                f"Found {len(duplicates)} duplicate rows. Examples: {example_ids}"
            )
            
            logger.error(err_msg)
            
            raise ValueError(err_msg)

    def _check_obligatory_columns(self, df: pd.DataFrame) -> None:
        """
        Ensures all required identity and feature columns exist.
        """
        required_cols = list(set(self.identity_cols + self.config["features"]))
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            err_msg = (
                f"\n[CRITICAL DATA ERROR] DataSniffer: Missing Obligatory Columns!\n"
                f"Missing: {missing_cols}\n"
                f"Available: {df.columns.tolist()}"
            )
            
            logger.error(err_msg)
            
            raise ValueError(err_msg)

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

            err_msg = (
                "\n[CRITICAL DATA ERROR] DataSniffer detected non-finite values!\n"
                "Offending Columns:\n" + "\n".join(error_details)
            )
            
            logger.error(err_msg)
            
            raise ValueError(err_msg)

    def sniff_pure_state_parity(self, df_input: pd.DataFrame, df_output: pd.DataFrame) -> None:
        """
        Verifies that the output DataFrame is bit-wise identical to the input DataFrame
        once model predictions are stripped.
        """
        print("")
        logger.info("DataSniffer: Starting Pure State Parity Audit")

        time_col = self.config["time_col"]
        id_col = self.config["id_col"]

        # 1. Strip predictions from output
        pred_cols = [c for c in df_output.columns if c.startswith("pred_")]
        generated_binary = [c for c in df_output.columns if c.startswith("by_") and c not in df_input.columns]

        df_stripped = df_output.drop(columns=pred_cols + generated_binary)

        # 2. Alignment
        if isinstance(df_input.index, pd.MultiIndex):
            df_in_aligned = df_input.reset_index()
        else:
            df_in_aligned = df_input.copy()

        df_out_aligned = df_stripped.reset_index()

        # 3. Structural Integrity (Columns must match)
        df_out_aligned = df_out_aligned[df_in_aligned.columns]

        # 4. Type-Agnostic Comparison (Force stripped to match input types)
        for col in df_in_aligned.columns:
            df_out_aligned[col] = df_out_aligned[col].astype(df_in_aligned[col].dtype)

        # 5. Order-Agnostic Comparison (Sort both by Time and ID)
        df_in_aligned = df_in_aligned.sort_values(by=[time_col, id_col]).reset_index(drop=True)
        df_out_aligned = df_out_aligned.sort_values(by=[time_col, id_col]).reset_index(drop=True)

        # 6. Robust Comparison (Structural + AllClose for Floats)
        drift_report = []
        for col in df_in_aligned.columns:
            if pd.api.types.is_float_dtype(df_in_aligned[col]):
                # Use np.allclose for floats
                if not np.allclose(df_in_aligned[col].values, df_out_aligned[col].values, atol=1e-5):
                    drift_report.append(f" - Column '{col}': Float values drifted beyond tolerance.")
            else:
                # Use standard equality for IDs and other types
                if not df_in_aligned[col].equals(df_out_aligned[col]):
                    drift_report.append(f" - Column '{col}': Categorical/Integer drift detected.")

        if drift_report:
            err_msg = (
                "[CRITICAL DATA ERROR] DataSniffer: Pure State Parity Violation!\n"
                "Output (minus predictions) has drifted from Input.\n"
                + "\n".join(drift_report)
            )
            
            logger.error(err_msg)
            
            raise ValueError(err_msg)

        logger.info("DataSniffer: Pure State Parity Audit Passed.")

    def sniff_pure_state_schema(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        """
        Enforces the ADR 032 structural contract on the output DataFrame.
        """
        print("")
        logger.info("DataSniffer: Starting Pure State Schema Audit")

        # 1. MultiIndex Check (ADR 032 Section 2.1)
        expected_index = [config["time_col"], config["id_col"]]
        if list(df.index.names) != expected_index:
            err_msg = f"DataSniffer: Index Mismatch! Expected {expected_index}, got {list(df.index.names)}"
            
            logger.error(err_msg)
            
            raise ValueError(err_msg)

        # 2. Identity Column Check (ADR 032 Section 2.2)
        # c_id is non-negotiable
        mandatory = ["c_id"] + list(config["spatial_cols"])
        missing = [c for c in mandatory if c not in df.columns]
        if missing:
            err_msg = f"DataSniffer: Missing Mandatory Identities! {missing}"
            
            logger.error(err_msg)
            
            raise ValueError(err_msg)

        # 3. Target Prefix Check (ADR 032 Section 2.4)
        targets = config["features"]
        for target in targets:
            # We expect pred_lr_{target} and pred_by_{target}
            # Note: target itself might already have lr_ prefix in some configs
            # We assume features in config are the base names or lr_ prefixed names.
            # To be safe, we check for columns starting with pred_ and containing the target
            found_preds = [c for c in df.columns if c.startswith("pred_") and target in c]
            if not found_preds:
                err_msg = f"DataSniffer: Missing Predictions for target '{target}'"
                
                logger.error(err_msg)
                
                raise ValueError(err_msg)

            # Verify retirement of suffixes
            offending = [c for c in found_preds if c.endswith("_raw") or c.endswith("_prob")]
            if offending:
                err_msg = f"DataSniffer: Legacy Suffixes detected! {offending} violates ADR 032."
                
                logger.error(err_msg)
                
                raise ValueError(err_msg)

        logger.info("DataSniffer: Pure State Schema Audit Passed.")
