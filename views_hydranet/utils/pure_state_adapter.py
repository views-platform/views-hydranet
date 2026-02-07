"""
PureStateAdapter: Formalizes the ADR 032 "Pure State" Output Schema.
Governed by ADR 040 (The Subsetting Gate).
"""

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

class PureStateAdapter:
    """
    Authoritative actor for enforcing the HydraNet output contract.
    
    Translates "Dirty" internal DataFrames into the 12-feature "Pure State" 
    representation required by the ViEWS pipeline.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes with target and identity metadata.
        """
        self.targets = config.get("targets", [])
        self.classification_outputs = config.get("classification_outputs", [])
        self.identity_cols = ["c_id", "row", "col"] # ADR 032 Mandatory Identity Anchors

    def enforce_pure_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subsets and renames columns to satisfy ADR 032.
        
        Rules:
        1. Keep Mandatory Identities (c_id, row, col).
        2. Keep Actuals (lr_ targets and their by_ derivatives).
        3. Keep Predictions (pred_lr_ and pred_by_).
        4. Drop all other internal/intermediate channels.
        """
        if df is None or df.empty:
            return df

        final_cols = []
        
        # 1. Identity Handshake
        for col in self.identity_cols:
            if col in df.columns:
                final_cols.append(col)
            else:
                 logger.debug(f"PureStateAdapter: Identity '{col}' missing from inbound DataFrame.")

        # 2. Target Handshake (ADR 032 / 033 Symmetry)
        for t in self.targets:
            if not t.startswith("lr_"):
                # Law 6 Violation
                raise ValueError(f"PureStateAdapter Contract Violation: Target '{t}' must start with 'lr_' prefix.")

            # Derive the 4-Column Block for this target
            binary_t = t.replace("lr_", "by_", 1)
            pred_lr_t = f"pred_{t}"
            pred_by_t = f"pred_{binary_t}"

            block = [t, binary_t, pred_lr_t, pred_by_t]
            
            for col in block:
                if col in df.columns:
                    final_cols.append(col)
                else:
                    logger.debug(f"PureStateAdapter: Target component '{col}' missing from inbound DataFrame.")

        # 3. Final Subsetting
        df_pure = df[final_cols].copy()
        
        logger.info(f"✨ PureStateAdapter: Contract satisfied ({len(final_cols)} features).")
        return df_pure

    def enforce_pure_state_list(self, list_df: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Convenience wrapper for lists of DataFrames (Backtests)."""
        return [self.enforce_pure_state(df) for df in list_df]
