import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, ANY
from views_hydranet.manager.hydranet_manager import HydranetManager
from pathlib import Path

def test_orchestration_loop_indices():
    """
    Verify the logic of the rolling origin loop and target selection.
    """
    # 1. Setup
    # Use a pure MagicMock for the manager to avoid ALL pipeline core logic
    manager = MagicMock()
    manager.config = {
        "run_type": "validation",
        "time_steps": 36,
        "test_samples": 1,
        "input_channels": 3,
        "target_variable": "ns",
        "targets": ["ln_sb_best"] # Add target to trigger loop
    }
    manager.configs = manager.config # Also set configs for target selection loop
    manager.device = "cpu"
    manager._model_path = MagicMock()
    # Ensure directory exists for file writing in mock
    gen_path = Path("/tmp/gen")
    gen_path.mkdir(parents=True, exist_ok=True)
    manager._model_path.data_generated = gen_path
    manager._load_model_artifact.return_value = (MagicMock(), "ts")

    with patch("views_hydranet.manager.hydranet_manager.create_or_load_views_vol") as mock_vol_loader, \
         patch("views_hydranet.manager.hydranet_manager.HydraNetInference") as mock_inference_cls, \
         patch("views_hydranet.manager.hydranet_manager.zstack_to_contract_df") as mock_contract_conv, \
         patch("views_hydranet.manager.hydranet_manager.pickle.dump"), \
         patch("views_hydranet.manager.hydranet_manager.validate_contract_dataframes"):
        
        mock_vol_loader.return_value = np.zeros((48, 10, 10, 8))
        
        mock_inference = mock_inference_cls.return_value
        mock_inference.generate_posterior_samples.return_value = (np.zeros((36, 10, 10, 6, 1)), np.zeros((36, 10, 10, 8, 1)))
        mock_contract_conv.return_value = [pd.DataFrame()]
        
        # 2. Execute
        # Directly call the method using the Mock instance as 'self'
        results = HydranetManager._evaluate_model_artifact(manager, eval_type="offline")
        
        # 3. Assertions
        # 12 windows * 1 target = 12 DFs
        assert len(results) == 12
        
        # Verify that the correct target was passed to the converter
        mock_contract_conv.assert_called_with(
            posterior_zstack=ANY,
            meta_zstack=ANY,
            target="ln_sb_best"
        )
        
        # Verify orchestration indices logic
        # origins should be range(12) for 48 months and 36 steps
        first_call_args = mock_inference.generate_posterior_samples.call_args_list[0]
        assert first_call_args[0][0].shape[0] == 37
