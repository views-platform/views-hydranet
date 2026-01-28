import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from views_hydranet.manager.hydranet_manager import HydranetManager
from pathlib import Path

class MockManager(HydranetManager):
    """
    Subclass to bypass pipeline core init.
    """
    def __init__(self):
        self.config = {
            "run_type": "validation",
            "time_steps": 36,
            "test_samples": 1,
            "input_channels": 3
        }
        self.device = "cpu"
        self._model_path = MagicMock()
        self._model_path.data_generated = Path("/tmp/gen")

def test_orchestration_loop_indices():
    """
    Verify the logic of the rolling origin loop.
    """
    # 1. Setup
    manager = MockManager()
    
    with patch("views_hydranet.manager.hydranet_manager.create_or_load_views_vol") as mock_vol_loader, \
         patch("views_hydranet.manager.hydranet_manager.HydraNetInference") as mock_inference_cls, \
         patch("views_hydranet.manager.hydranet_manager.zstack_to_contract_df") as mock_contract_conv, \
         patch("views_hydranet.manager.hydranet_manager.pickle.dump"):
        
        # 48 months: origin 0..11, each followed by 36 months?
        # Wait, if we have 48 months and TS=36.
        # Last origin must have 36 months following it.
        # 48 - 36 - 1 = 11.
        # Origins would be 0, 1, ..., 11. Total 12. Perfect.
        mock_vol_loader.return_value = np.zeros((48, 10, 10, 8))
        manager._load_model_artifact = MagicMock(return_value=(MagicMock(), "ts"))
        
        mock_inference = mock_inference_cls.return_value
        mock_inference.generate_posterior_samples.return_value = (np.zeros((36, 10, 10, 6, 1)), np.zeros((36, 10, 10, 8, 1)))
        mock_contract_conv.return_value = [pd.DataFrame()]
        
        # 2. Execute
        results = manager._evaluate_model_artifact(eval_type="offline")
        
        # 3. Assertions
        assert len(results) == 12
        assert mock_inference.generate_posterior_samples.call_count == 12
        
        # Verify slicing logic: first call should have origin index slice
        # first origin is (48-36-1) - 11 = 0.
        # Slice should be vol[: 0 + 1 + 36] -> vol[:37]
        first_call_args = mock_inference.generate_posterior_samples.call_args_list[0]
        sliced_vol = first_call_args[0][0]
        assert sliced_vol.shape[0] == 37
        
        # Last origin is 11.
        # Slice should be vol[: 11 + 1 + 36] -> vol[:48]
        last_call_args = mock_inference.generate_posterior_samples.call_args_list[-1]
        sliced_vol = last_call_args[0][0]
        assert sliced_vol.shape[0] == 48
