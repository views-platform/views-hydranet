import pandas as pd
import pytest

# C-138/F-Z1: guard the SUBMODULE actually imported below, not just the top package. The installed
# views_evaluation top package imports fine, but `evaluation.evaluation_manager` was removed
# upstream — a top-level importorskip is a no-op and collection hard-errors on the import. Guarding
# the submodule makes this file SKIP cleanly (so a plain `pytest` collects without --ignore).
pytest.importorskip("views_evaluation.evaluation.evaluation_manager")

from views_evaluation.evaluation.evaluation_manager import EvaluationManager


def test_eval_package_contract_acceptance():
    """
    STRICT INTEGRATION TEST:
    Verifies that the REAL views-evaluation package accepts our 'Contract DataFrame'.
    """

    # 1. Define the Metric Config
    # Update: EvaluationManager.__init__ takes no arguments.
    # Metrics are likely inferred or passed via config if needed.
    eval_manager = EvaluationManager()

    # 2. Construct 'Toy Actuals' (Raw Scale)
    # The evaluation library expects a MultiIndex actuals too for processed data
    df_actual = pd.DataFrame(
        {
            "month_id": [400, 400, 401, 401],
            "priogrid_gid": [1, 2, 1, 2],
            "lr_ged_sb": [10.0, 50.0, 12.0, 60.0],
        }
    ).set_index(["month_id", "priogrid_gid"])

    # 3. Construct 'Toy Predictions' (Our Output)
    df_predictions = pd.DataFrame(
        {
            "month_id": [400, 400, 401, 401],
            "priogrid_gid": [1, 2, 1, 2],
            "pred_lr_ged_sb": [[9.0, 11.0], [48.0, 52.0], [11.0, 13.0], [58.0, 62.0]],
        }
    ).set_index(["month_id", "priogrid_gid"])

    # 4. Mock pipeline configs (eval lib needs steps as a LIST of ints)
    # Add metrics here just in case
    pipeline_configs = {
        "targets": ["lr_ged_sb"],
        "run_type": "validation",
        "steps": [1],
        "regression_metrics": ["MSE"],
        "classification_metrics": ["AP"],
        "regression_point_metrics": ["MSE"],
        "classification_point_metrics": ["AP"],
    }

    # 5. EXECUTE THE REAL EVALUATION
    try:
        results = eval_manager.evaluate(df_actual, [df_predictions], "lr_ged_sb", pipeline_configs)

        assert "step" in results
        assert "time_series" in results
        assert "month" in results
        print("\n✅ Evaluation Package accepted the aligned Toy DataFrame!")

    except Exception as e:
        print("\n❌ Evaluation Package REJECTED the Toy DataFrame!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        raise e
