
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

def check_stochastic_type():
    config = {
        'time_col': 'month_id',
        'id_col': 'pg_id',
        'spatial_cols': ['row', 'col'],
        'identity_cols': ['month_id', 'pg_id'],
        'row_offset': 0,
        'col_offset': 0
    }
    df = pd.DataFrame({
        'month_id': [1], 'pg_id': [100], 'row': [0], 'col': [0], 'feat': [0.5]
    })
    handler = VolumeHandler.from_df(df, config, height=2, width=2)
    
    # 1 Time, 2 Height, 2 Width, 2 Channels (1 Signal, 1 Prob), 5 Samples
    posterior = np.random.rand(1, 2, 2, 2, 5).astype(np.float32)
    
    pred_handler = handler.wrap_predictions(posterior, base_names=['target'])
    df_out = pred_handler.to_evaluation_df(history=handler, start_idx=0)
    
    print("Dtypes:")
    print(df_out.dtypes)
    print("\nIndex Dtypes:")
    print(df_out.index.dtype if not isinstance(df_out.index, pd.MultiIndex) else [level.dtype for level in df_out.index.levels])
    print(f"Series Dtype: {df_out['pred_target_raw'].dtype}")
    print(f"Cell Type: {type(cell)}")
    print(f"Cell Content: {cell}")
    print(f"Is list? {isinstance(cell, list)}")
    
    # Check if we can convert to list easily if needed
    if not isinstance(cell, list):
        try:
            as_list = cell.tolist()
            print(f"Converted to list: {as_list}")
        except:
            print("Could not convert to list via .tolist()")

if __name__ == "__main__":
    check_stochastic_type()
