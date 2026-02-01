import logging

import pandas as pd
from views_evaluation.evaluation.evaluation_manager import EvaluationManager

# Suppress warnings for cleaner output
logging.getLogger("views_evaluation").setLevel(logging.ERROR)

def probe(target_name, actual_val, pred_val):
    print(f"\n--- Probing target: {target_name} ---")
    idx = pd.MultiIndex.from_tuples([(500, 1)], names=['month_id', 'priogrid_gid'])
    # Actuals as provided
    actuals = pd.DataFrame({target_name: [actual_val]}, index=idx)
    # Prediction as provided (wrapped in list for canonical format)
    preds = [pd.DataFrame({f"pred_{target_name}": [[pred_val]]}, index=idx)]

    manager = EvaluationManager(metrics_list=["MSE"])
    results = manager.evaluate(actuals, preds, target_name, config={"steps": [1]})
    mse = results["step"][1].loc["step01", "MSE"]
    print(f"  Actual input: {actual_val}")
    print(f"  Pred input:   {pred_val}")
    print(f"  Resulting MSE: {mse:.6f}")

# Test 1: 'lr' prefix (Raw)
# Expected: MSE = (100 - 4.61512)^2 = 9098.37
probe("lr_sb", 100.0, 4.61512)

# Test 2: 'ln' prefix (Log)
# Both actual and pred are logged (ln(100+1) ~ 4.61512)
# Expected: Library unlogs both to 100, MSE = 0
probe("lr_sb", 4.61512, 4.61512)

# Test 3: Mixed (This is the "Silent Error" risk)
# Target is 'lr' but we give it logs
probe("lr_sb", 4.61512, 4.61512)
