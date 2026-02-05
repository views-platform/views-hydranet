
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

def test_classification_and_raw_suffixes():
    # Setup minimal config
    config = {
        'time_col': 'month_id',
        'id_col': 'pg_id',
        'spatial_cols': ['row', 'col'],
        'identity_cols': ['month_id', 'pg_id'],
        'row_offset': 0,
        'col_offset': 0
    }
    
    # Create dummy data
    df = pd.DataFrame({
        'month_id': [1],
        'pg_id': [100],
        'row': [0],
        'col': [0],
        'feat': [0.5]
    })
    
    # Initialize Handler
    config["height"], config["width"] = 2, 2
    handler = VolumeHandler.from_df(df, config)
    
    # Simulate posterior: 1 Batch, 1 Time, 2 Channels (1 Signal, 1 Prob), 2 Height, 2 Width
    # We provide 1 base name "target", so we expect 2 output channels: target_INTERNAL_SIGNAL, target_INTERNAL_PROB
    posterior = torch.zeros((1, 1, 2, 2, 2)) 
    
    # Wrap predictions
    pred_handler = handler.wrap_predictions(posterior, base_names=['target'])
    
    # Convert to DataFrame
    df_out = pred_handler.to_evaluation_df(history=handler, start_idx=0)
    
    # Assertions
    assert "pred_target_raw" in df_out.columns, "Missing regression output (pred_target_raw)"
    assert "pred_target_prob" in df_out.columns, "Missing classification output (pred_target_prob)"
