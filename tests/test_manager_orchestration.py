import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from views_hydranet.manager.hydranet_manager import HydranetManager
from pathlib import Path

def test_evaluate_model_artifact_orchestration():
    """
    Verify that _evaluate_model_artifact produces 12 sequences for validation.
    """
    # 1. Mock setup
    mock_path = MagicMock()
    mock_path.artifacts = Path("/tmp/artifacts")
    mock_path.data_processed = Path("/tmp/data_processed")
    mock_path.data_raw = Path("/tmp/data_raw")
    mock_path.data_generated = Path("/tmp/data_generated")
    
    with patch("views_hydranet.manager.hydranet_manager.create_or_load_views_vol") as mock_vol_loader, \
         patch("views_hydranet.manager.hydranet_manager.HydraNetInference") as mock_inference_cls, \
         patch("views_hydranet.manager.hydranet_manager.zstack_to_contract_df") as mock_contract_conv:
        
        # Mock 48 month volume (12 rolling windows of 36 steps each)
        # Shape: [months, H, W, channels]
        mock_vol_loader.return_value = np.zeros((48, 180, 180, 8))
        
        # Mock Manager config
        manager = HydranetManager(model_path=mock_path)
        manager.config = {
            "run_type": "validation",
            "time_steps": 36,
            "test_samples": 1,
            "input_channels": 3
        }
        
        # Mock model loading
        manager._load_model_artifact = MagicMock(return_value=(MagicMock(), "20260128_120000"))
        
        # Mock inference results
        mock_inference = mock_inference_cls.return_value
        mock_inference.generate_posterior_samples.return_value = (np.zeros((36, 180, 180, 6, 1)), np.zeros((36, 180, 180, 8, 1)))
        
        # Mock contract conversion to return a single DF list per call
        mock_contract_conv.return_value = [pd.DataFrame({"pred_lr_sb": [[0.0]]})]
        
        # 2. Execute
        results = manager._evaluate_model_artifact(eval_type="offline")
        
        # 3. Assertions
        assert isinstance(results, list)
        # STANDARD: 12 sequences for offline evaluation
        assert len(results) == 12, f"Expected 12 sequences, got {len(results)}"
        
        # Verify inference was called 12 times
        assert mock_inference.generate_posterior_samples.call_count == 12
