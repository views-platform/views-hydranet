import numpy as np
import pandas as pd
import pytest
from views_hydranet.utils.utils_contract_converters import zstack_to_contract_df
from views_evaluation.evaluation.evaluation_manager import EvaluationManager

def test_golden_metric_regression():
    """
    GOLDEN REGRESSION TEST:
    Ensures that a fixed input always produces the exact same evaluation metrics.
    This protects against silent logic changes in the producer-consumer chain.
    """
    # 1. Setup Deterministic Mock Data
    np.random.seed(42)
    steps, samples, H, W = 2, 10, 4, 4
    
    # Random magnitudes in log-space [0, 2]
    posterior_zstack = np.random.uniform(0, 2, (steps, H, W, 3, samples))
    
    # Metadata zstack
    meta_zstack = np.zeros((steps, H, W, 8, 1))
    for t in range(steps):
        # pg_id 1..16
        meta_zstack[t, :, :, 0, 0] = np.arange(1, H*W + 1).reshape(H, W)
        meta_zstack[t, :, :, 3, 0] = 500 + t
        
    # 2. Transform to Contract DF
    target = "sb"
    list_df_predictions = zstack_to_contract_df(posterior_zstack, meta_zstack, target)
    
    # 3. Create Deterministic Actuals
    # Generate actuals that are somewhat related to predictions but with noise
    actuals_rows = []
    for t in range(steps):
        month_id = 500 + t
        for pg_id in range(1, H*W + 1):
            # Deterministic noise based on pg_id and t
            val = (pg_id % 5) + (t * 2)
            actuals_rows.append({
                "month_id": month_id,
                "priogrid_gid": pg_id,
                f"lr_{target}": float(val)
            })
    actuals_df = pd.DataFrame(actuals_rows).set_index(["month_id", "priogrid_gid"])
    
    # 4. Run Real Evaluation
    metrics = ["CRPS"] # MSE skipped for uncertainty tasks in this version
    manager = EvaluationManager(metrics_list=metrics)
    eval_config = {"steps": list(range(1, steps + 1))}
    
    results = manager.evaluate(
        actual=actuals_df,
        predictions=list_df_predictions,
        target=f"lr_{target}",
        config=eval_config
    )
    
    # 5. Extract Step-Wise Results
    step_results_df = results["step"][1]
    print(f"\nDEBUG: Available Columns: {step_results_df.columns.tolist()}")
    
    # --- GOLDEN VALUES (Capture Mode) ---
    actual_crps_s1 = step_results_df.loc["step01", "CRPS"]
    actual_crps_s2 = step_results_df.loc["step02", "CRPS"]
    
    print(f"DEBUG: Actual CRPS Step 1: {actual_crps_s1:.6f}")
    print(f"DEBUG: Actual CRPS Step 2: {actual_crps_s2:.6f}")
    
    # Baseline values captured for seed 42
    assert pytest.approx(actual_crps_s1, abs=1e-6) == 0.929542
    assert pytest.approx(actual_crps_s2, abs=1e-6) == 1.645735


