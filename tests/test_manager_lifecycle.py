import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from views_hydranet.manager.hydranet_manager import HydranetManager
from pathlib import Path

@pytest.fixture
def manager_env(tmp_path):
    """
    Sets up a real physical environment for testing the manager lifecycle.
    """
    # 1. Setup paths
    raw_dir = tmp_path / "data" / "raw"
    art_dir = tmp_path / "artifacts"
    raw_dir.mkdir(parents=True)
    art_dir.mkdir(parents=True)
    
    # 2. Create dummy actuals
    df_path = raw_dir / "validation_viewser_df.parquet"
    df = pd.DataFrame({
        "month_id": [100, 100],
        "priogrid_gid": [1, 2],
        "ln_sb_best": [1.0, 2.0]
    })
    df.to_parquet(df_path)
    
    # 3. Create dummy log file (for mirroring check)
    log_path = raw_dir / "validation_data_fetch_log.txt"
    log_path.write_text("Fetched at 2026-01-29")
    
    # 4. Mock the Manager
    with patch("views_hydranet.manager.hydranet_manager.HydranetManager.__init__", return_value=None), \
         patch("views_hydranet.manager.hydranet_manager.setup_device", return_value="cpu"):
        
        m = HydranetManager(model_path=MagicMock())
        m._model_path = MagicMock()
        m._model_path.data_raw = raw_dir
        m._model_path.artifacts = art_dir
        m.device = "cpu"
        
        # Mock managed configs
        m._config_dict = {
            "run_type": "validation",
            "targets": ["ln_sb_best"],
            "target_variable": "sb"
        }
        type(m).configs = property(lambda self: self._config_dict, 
                                   lambda self, v: self._config_dict.update(v))
        type(m).config = property(lambda self: self._config_dict)
        
        return m, raw_dir, art_dir

def test_manager_evaluation_lifecycle_explicit_augmentation(manager_env):
    """
    Verify the full explicit flow: Read -> Augment -> Save Shadow -> Mirror -> Redirect.
    """
    manager, raw_dir, art_dir = manager_env
    shadow_dir = art_dir / "tmp_eval_data"
    
    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager._execute_model_evaluation") as mock_super:
        
        # Assert state DURING the call
        def side_effect():
            # 1. Check path redirection
            assert manager._model_path.data_raw == shadow_dir
            # 2. Check file presence in shadow
            assert (shadow_dir / "validation_viewser_df.parquet").exists()
            # 3. Check mirroring (log exists in shadow)
            assert (shadow_dir / "validation_data_fetch_log.txt").exists()
            # 4. Check augmentation (unlogged column exists)
            df_aug = pd.read_parquet(shadow_dir / "validation_viewser_df.parquet")
            assert "lr_sb_best" in df_aug.columns
            
        mock_super.side_effect = side_effect
        
        # Execute
        manager._execute_model_evaluation()
        
    # Verify Cleanup
    assert not shadow_dir.exists()
    assert manager._model_path.data_raw == raw_dir

def test_manager_restoration_under_chaos(manager_env):
    """
    Prove that paths are restored even if the core logic crashes.
    """
    manager, raw_dir, _ = manager_env
    
    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager._execute_model_evaluation", 
               side_effect=RuntimeError("Chaos")):
        
        with pytest.raises(RuntimeError, match="Chaos"):
            manager._execute_model_evaluation()
            
    # CRITICAL: Redirection must be reversed
    assert manager._model_path.data_raw == raw_dir
