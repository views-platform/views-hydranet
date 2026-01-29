import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, ANY
from views_hydranet.manager.hydranet_manager import HydranetManager
from pathlib import Path

def test_orchestration_loop_indices():
    """
    Verify the logic of the rolling origin loop.
    We test _evaluate_model_artifact using a mock manager that satisfies all calls.
    """
    # 1. Setup a mock manager that behaves like the real one
    manager = MagicMock()
    manager.config = {
        "run_type": "validation",
        "time_steps": 3,
        "test_samples": 1,
        "input_channels": 3,
        "target_variable": "ns"
    }
    manager.configs = {"targets": ["lr_ns_best"]}

    # Critical: Mock internal methods to return expected values
    manager._load_model_artifact.return_value = (MagicMock(), "20260129_120000")
    manager._translate_targets.return_value = ["lr_ns_best"]

    # Path mocking
    manager._model_path = MagicMock()
    manager._model_path.data_generated = Path("/tmp/gen")
    manager._model_path.data_generated.mkdir(parents=True, exist_ok=True)

    # 2. Setup external patches
    with patch("views_hydranet.manager.hydranet_manager.create_or_load_views_vol") as mock_vol_loader:
        with patch("views_hydranet.manager.hydranet_manager.HydraNetInference") as mock_inference_cls:
            with patch("views_hydranet.manager.hydranet_manager.zstack_to_contract_df") as mock_contract_conv:
                with patch("views_hydranet.manager.hydranet_manager.pickle.dump"):
                    with patch("views_hydranet.manager.hydranet_manager.validate_contract_dataframes"):

                        mock_vol_loader.return_value = np.zeros((15, 10, 10, 8))
                        mock_inference = mock_inference_cls.return_value
                        mock_inference.generate_posterior_samples.return_value = (
                            np.zeros((3, 10, 10, 6, 1)),
                            np.zeros((3, 10, 10, 8, 1))
                        )
                        # Ensure the converter returns a list with one DF
                        mock_contract_conv.return_value = [pd.DataFrame({"pred_lr_ns_best": [0.5]})]

                        # 3. Execute the REAL method on the MOCK instance
                        results = HydranetManager._evaluate_model_artifact(manager, eval_type="offline")

                        # 4. Assertions
                        # Check that the converter was called with the target in config
                        mock_contract_conv.assert_any_call(
                            posterior_zstack=ANY,
                            meta_zstack=ANY,
                            target="lr_ns_best",
                            config=manager.config
                        )
