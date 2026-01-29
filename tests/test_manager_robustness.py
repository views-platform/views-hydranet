import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, ANY
import views_pipeline_core.files.utils as utils_module
from views_hydranet.manager.hydranet_manager import HydranetManager
from pathlib import Path

@pytest.fixture
def robust_manager():
    """Returns a manager mock that satisfies property-based config access."""
    m = MagicMock(spec=HydranetManager)
    m.device = "cpu"
    m._config_dict = {
        "run_type": "validation",
        "time_steps": 3,
        "test_samples": 1,
        "input_channels": 3,
        "target_variable": "sb",
        "targets": ["ln_sb_best"]
    }
    # Simulate the managed properties
    type(m).configs = property(lambda self: self._config_dict, 
                               lambda self, v: self._config_dict.update(v))
    type(m).config = property(lambda self: self._config_dict)
    
    m._model_path = MagicMock()
    m._model_path.data_generated = Path("/tmp/gen_robust")
    m._model_path.data_generated.mkdir(parents=True, exist_ok=True)
    return m

def test_multitask_merging_alignment(robust_manager):
    """
    PROVE the fix for the Flattening Trap:
    Multiple targets should result in exactly ONE DataFrame per origin with multiple columns.
    """
    robust_manager._config_dict["targets"] = ["lr_sb_best", "lr_ns_best"]
    robust_manager._config_dict["target_variable"] = "" # No filter
    robust_manager._load_model_artifact.return_value = (MagicMock(), "ts")
    robust_manager._translate_targets.return_value = ["lr_sb_best", "lr_ns_best"]
    
    with patch("views_hydranet.manager.hydranet_manager.create_or_load_views_vol") as mock_vol:
        with patch("views_hydranet.manager.hydranet_manager.HydraNetInference") as mock_inf_cls:
            with patch("views_hydranet.manager.hydranet_manager.zstack_to_contract_df") as mock_conv:
                with patch("views_hydranet.manager.hydranet_manager.pickle.dump"):
                    with patch("views_hydranet.manager.hydranet_manager.validate_contract_dataframes"):
        
                        mock_vol.return_value = np.zeros((10, 10, 10, 8))
                        
                        # Fix the Unpacking Error here
                        mock_inference = mock_inf_cls.return_value
                        mock_inference.generate_posterior_samples.return_value = (MagicMock(), MagicMock())
                        
                        def side_effect(posterior_zstack, meta_zstack, target):
                            return [pd.DataFrame({f"pred_{target}": [0.5]}, index=pd.MultiIndex.from_tuples([(1,1)], names=["month_id", "priogrid_gid"]))]
                        mock_conv.side_effect = side_effect
                        
                        results = HydranetManager._evaluate_model_artifact(robust_manager, eval_type="offline")
                        
                        # 10 months, 3 steps -> 7 origins
                        assert len(results) == 7
                        assert "pred_lr_sb_best" in results[0].columns
                        assert "pred_lr_ns_best" in results[0].columns
                        assert results[0].shape[1] == 2

def test_manager_restoration_under_chaos(robust_manager):
    """
    PROVE the Global State Protection:
    The redirection MUST be reversed even if an unexpected exception occurs.
    """
    original_raw_path = robust_manager._model_path.data_raw
    robust_manager._config_dict["targets"] = ["ln_sb_best"]
    robust_manager._translate_targets.return_value = ["lr_sb_best"]
    
    with patch("views_hydranet.manager.hydranet_manager.read_dataframe", return_value=pd.DataFrame({"ln_sb_best": [1.0]})):
        with patch("views_hydranet.manager.hydranet_manager.save_dataframe"):
            with patch("views_pipeline_core.managers.model.model.ForecastingModelManager._execute_model_evaluation", 
                       side_effect=RuntimeError("Chaos")):
                with patch("os.remove"):
                    with patch("os.rmdir"):
                        with patch("os.listdir", return_value=[]):
                            with patch("pathlib.Path.mkdir"):
                
                                with pytest.raises(RuntimeError, match="Chaos"):
                                    HydranetManager._execute_model_evaluation(robust_manager)
            
    assert robust_manager._model_path.data_raw == original_raw_path
    assert robust_manager.configs["targets"] == ["ln_sb_best"]

def test_partition_aware_windows(robust_manager):
    """Verify that forecasting partition correctly defaults to 1 window."""
    robust_manager._config_dict["run_type"] = "forecasting"
    robust_manager._load_model_artifact.return_value = (MagicMock(), "ts")
    robust_manager._translate_targets.return_value = ["lr_sb_best"]
    
    with patch("views_hydranet.manager.hydranet_manager.create_or_load_views_vol") as mock_vol:
        with patch("views_hydranet.manager.hydranet_manager.HydraNetInference") as mock_inf_cls:
            with patch("views_hydranet.manager.hydranet_manager.zstack_to_contract_df", return_value=[pd.DataFrame()]):
                with patch("views_hydranet.manager.hydranet_manager.pickle.dump"):
                    with patch("views_hydranet.manager.hydranet_manager.validate_contract_dataframes"):
        
                        mock_vol.return_value = np.zeros((100, 10, 10, 8))
                        mock_inference = mock_inf_cls.return_value
                        mock_inference.generate_posterior_samples.return_value = (MagicMock(), MagicMock())
                        
                        results = HydranetManager._evaluate_model_artifact(robust_manager, eval_type="offline")
                        assert len(results) == 1