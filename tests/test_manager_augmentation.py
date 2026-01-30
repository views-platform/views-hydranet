import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from views_hydranet.manager.hydranet_manager import HydranetManager

@pytest.fixture
def clean_manager():
    """Returns a real manager instance with base init bypassed."""
    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None), \
         patch("views_hydranet.manager.hydranet_manager.setup_device", return_value="cpu"):
        
        m = HydranetManager(model_path=MagicMock())
        # Provide minimal config to satisfy handshake-free property access
        m._hydranet_config = {"some": "config"} 
        return m

def test_translate_targets(clean_manager):
    inputs = ["ln_sb_best", "ns_best", "lr_os_best"]
    expected = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
    assert clean_manager._translate_targets(inputs) == expected

def test_augment_dataframe_unlogging(clean_manager):
    df = pd.DataFrame({"ln_sb_best": [4.61512051681]})
    requested = ["lr_sb_best"]
    augmented = clean_manager._augment_dataframe(df, requested)
    assert "lr_sb_best" in augmented.columns
    assert np.allclose(augmented["lr_sb_best"], [100.0])

def test_augment_dataframe_binarization_from_raw(clean_manager):
    df = pd.DataFrame({"lr_sb_best": [0.0, 5.5]})
    requested = ["lr_sb_best_binarized"]
    augmented = clean_manager._augment_dataframe(df, requested)
    assert "lr_sb_best_binarized" in augmented.columns
    assert list(augmented["lr_sb_best_binarized"]) == [0.0, 1.0]

