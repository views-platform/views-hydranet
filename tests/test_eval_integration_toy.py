import pytest
import pandas as pd
import numpy as np
from views_evaluation.evaluation.evaluation_manager import EvaluationManager

def test_eval_package_contract_acceptance():
    """
    STRICT INTEGRATION TEST:
    Verifies that the REAL views-evaluation package accepts our 'Contract DataFrame'.
    """
    
    # 1. Define the Metric Config
    metrics_config = ["mse", "mae"]
    eval_manager = EvaluationManager(metrics_config)
    
    # 2. Construct 'Toy Actuals' (Raw Scale)
    # The evaluation library expects a MultiIndex actuals too for processed data
    df_actual = pd.DataFrame({
        "month_id": [400, 400, 401, 401],
        "priogrid_gid": [1, 2, 1, 2],
        "lr_sb_best": [10.0, 50.0, 12.0, 60.0]
    }).set_index(["month_id", "priogrid_gid"])
    
    # 3. Construct 'Toy Predictions' (Our Output)
    df_predictions = pd.DataFrame({
        "month_id": [400, 400, 401, 401],
        "priogrid_gid": [1, 2, 1, 2],
        "pred_lr_sb_best": [
            [9.0, 11.0],
            [48.0, 52.0],
            [11.0, 13.0],
            [58.0, 62.0]
        ]
    }).set_index(["month_id", "priogrid_gid"])
    
    # 4. Mock pipeline configs (eval lib needs steps as a LIST of ints)
    pipeline_configs = {
        "targets": ["lr_sb_best"],
        "run_type": "validation",
        "steps": [1] 
    }
    
    # 5. EXECUTE THE REAL EVALUATION
    try:
        results = eval_manager.evaluate(
            df_actual, 
            [df_predictions], 
            "lr_sb_best", 
            pipeline_configs
        )
        
        assert "step" in results
        assert "time_series" in results
        assert "month" in results
        print("\n✅ Evaluation Package accepted the aligned Toy DataFrame!")
        
    except Exception as e:
        print(f"\n❌ Evaluation Package REJECTED the Toy DataFrame!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        raise e
