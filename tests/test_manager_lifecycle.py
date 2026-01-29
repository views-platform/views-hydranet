import pytest
from unittest.mock import MagicMock, patch
from views_hydranet.manager.hydranet_manager import HydranetManager
import views_pipeline_core.files.utils as utils_module

@pytest.fixture
def manager():
    with patch("views_hydranet.manager.hydranet_manager.HydranetManager.__init__", return_value=None):
        m = HydranetManager(model_path=MagicMock())
        # Mock the managed property
        m._config_dict = {"targets": ["ln_sb_best"]}
        type(m).configs = property(lambda self: self._config_dict, 
                                   lambda self, v: self._config_dict.update(v))
        return m

def test_manager_evaluation_lifecycle_state_restoration(manager):
    """
    Verify that the manager restores the targets config and 
    read_dataframe function even if the core evaluation fails.
    """
    original_read_df = utils_module.read_dataframe
    
    # Mock the base class to raise an error, simulating a core failure
    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager._execute_model_evaluation", 
               side_effect=RuntimeError("Core Failure")):
        
        with pytest.raises(RuntimeError, match="Core Failure"):
            manager._execute_model_evaluation()
            
    # 1. Verify Config Restoration
    assert manager.configs["targets"] == ["ln_sb_best"]
    
    # 2. Verify Monkey-Patch Restoration
    assert utils_module.read_dataframe == original_read_df

def test_manager_evaluation_intercepts_targets(manager):
    """
    Verify that the base class sees the translated 'lr_' targets.
    """
    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager._execute_model_evaluation") as mock_super:
        
        # We need to capture what self.configs looked like DURING the call
        # Since it's mutated and restored, we check the state in the mock call
        def check_config(*args, **kwargs):
            assert manager.configs["targets"] == ["lr_sb_best"]
            
        mock_super.side_effect = check_config
        
        manager._execute_model_evaluation()
        
    mock_super.assert_called_once()
    # Ensure it's restored after
    assert manager.configs["targets"] == ["ln_sb_best"]
