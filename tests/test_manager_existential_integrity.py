from unittest.mock import MagicMock, patch

from views_hydranet.manager.hydranet_manager import HydranetManager


def test_manager_methods_exist():
    """
    STRICT EXISTENTIAL TEST:
    Verifies that the manager class actually has the methods it claims to have.
    This prevents 'Indentation Traps' where methods become nested functions.
    """
    # 1. Arrange: Instantiate a REAL manager (mock only the __init__ side-effects)
    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None), \
         patch("views_hydranet.manager.hydranet_manager.setup_device", return_value="cpu"):

        manager = HydranetManager(model_path=MagicMock())

        # 2. Check for critical method names
        critical_methods = [
            "_load_model_artifact",
            "_execute_model_evaluation",
            "_execute_model_training",
            "_evaluate_model_artifact",
            "_translate_targets",
            "_augment_dataframe"
        ]

        for method_name in critical_methods:
            assert hasattr(manager, method_name), f"CRITICAL FAILURE: HydranetManager is missing method '{method_name}'! (Check indentation)"
            method = getattr(manager, method_name)
            assert callable(method), f"Method '{method_name}' exists but is not callable!"

        print("\n✅ All critical methods are present and correctly bound to the class.")
