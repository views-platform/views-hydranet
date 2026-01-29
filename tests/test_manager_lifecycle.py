import pytest
import os
import pandas as pd
from unittest.mock import MagicMock, patch
from views_hydranet.manager.hydranet_manager import HydranetManager
from pathlib import Path

@pytest.fixture
def manager():
    with patch("views_hydranet.manager.hydranet_manager.HydranetManager.__init__", return_value=None):
        m = HydranetManager(model_path=MagicMock())
        m._model_path = MagicMock()
        m._model_path.data_raw = Path("/tmp/raw")
        m._model_path.artifacts = Path("/tmp/art")
        
        # Mock the managed property
        m._config_dict = {"targets": ["ln_sb_best"], "run_type": "validation"}
        type(m).configs = property(lambda self: self._config_dict, 
                                   lambda self, v: self._config_dict.update(v))
        type(m).config = property(lambda self: self._config_dict)
        return m

def test_manager_evaluation_lifecycle_explicit_augmentation(manager):
    """
    Verify that the manager creates a shadow file and redirects the core.
    """
    with patch("views_hydranet.manager.hydranet_manager.read_dataframe", return_value=pd.DataFrame({"ln_sb_best": [1.0]})) as mock_read:
        with patch("views_hydranet.manager.hydranet_manager.save_dataframe") as mock_write:
            with patch("views_pipeline_core.managers.model.model.ForecastingModelManager._execute_model_evaluation") as mock_super:
                with patch("os.remove"):
                    with patch("os.rmdir"):
                        with patch("os.listdir", return_value=[]):
                            with patch("pathlib.Path.mkdir"):
        
                                # Capture the state DURING the call
                                def side_effect():
                                    assert manager._model_path.data_raw == manager._model_path.artifacts / "tmp_eval_data"
                                
                                mock_super.side_effect = side_effect
                                
                                # 1. Execute
                                manager._execute_model_evaluation()
                                
                                # 2. Verify Base Calls
                                mock_read.assert_called_once()
                                mock_write.assert_called_once()
                                mock_super.assert_called_once()
                                
                                # 3. Verify Restoration AFTER call
                                assert manager._model_path.data_raw == Path("/tmp/raw")

def test_manager_restoration_under_chaos(manager):
    """
    Ensure redirection is reversed even if core crashes.
    """
    with patch("views_hydranet.manager.hydranet_manager.read_dataframe", return_value=pd.DataFrame({"ln_sb_best": [1.0]})):
        with patch("views_hydranet.manager.hydranet_manager.save_dataframe"):
            with patch("views_pipeline_core.managers.model.model.ForecastingModelManager._execute_model_evaluation", side_effect=RuntimeError("Boom")):
                with patch("os.remove"):
                    with patch("os.rmdir"):
                        with patch("os.listdir", return_value=[]):
                            with patch("pathlib.Path.mkdir"):
        
                                with pytest.raises(RuntimeError, match="Boom"):
                                    manager._execute_model_evaluation()
            
    # CRITICAL: Path MUST be restored despite the RuntimeError
    assert manager._model_path.data_raw == Path("/tmp/raw")