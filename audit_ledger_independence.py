
import logging
import sys
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("AuditLedger")

def audit_custom_names():
    """
    H2.1: The Alias Audit.
    Does the system work with arbitrary column names if they are defined in the ledger?
    """
    logger.info("--- H2.1: Ledger Independence Audit (Custom Aliases) ---")

    # We rename EVERYTHING
    config = {
        "identity_cols": ["grid_id", "time_step", "y_coord", "x_coord", "task_id"],
        "features": ["violence_magnitude"],
        "row_offset": 0, "col_offset": 0, "height": 2, "width": 2,
        "steps": [1]
    }

    # We need to map 'row', 'col', 'month_id' internally for from_df 
    # OR we must prove that from_df is ALSO ledger-independent.
    # The current from_df uses hardcoded strings.
    
    data = [{
        "grid_id": 55, 
        "time_step": 100, 
        "y_coord": 1, 
        "x_coord": 1, 
        "task_id": 1, 
        "violence_magnitude": 0.7
    }]
    df_in = pd.DataFrame(data)

    logger.info("Probe: Attempting from_df with aliased columns.")
    try:
        # Note: from_df currently has hardcoded 'row', 'col', 'month_id' for indexing.
        # This test will likely fail at ingestion, proving Hypothesis 2 immediately.
        vh = VolumeHandler.from_df(df_in, config, height=2, width=2)
        logger.info("RESULT: from_df is surprisingly flexible (or I missed something).")
        
        logger.info("Probe: Attempting to_historical_df.")
        df_out = vh.to_historical_df()
        
        if "grid_id" in df_out.columns and df_out.iloc[0]["violence_magnitude"] == 0.7:
            logger.info("RESULT: Consistent with 'Spirit of Ledger'.")
        else:
            logger.error("RESULT: Falsified. Data scrambled or columns missing.")
            
    except Exception as e:
        logger.warning(f"BEHAVIOR: Execution failed with custom names: {e}")
        logger.info("RESULT: Falsified. Implementation is bound to hardcoded VIEWS names.")

if __name__ == "__main__":
    audit_custom_names()
