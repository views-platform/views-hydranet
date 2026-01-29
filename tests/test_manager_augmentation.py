import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from views_hydranet.manager.hydranet_manager import HydranetManager

@pytest.fixture
def manager():
    with patch("views_hydranet.manager.hydranet_manager.HydranetManager.__init__", return_value=None):
        m = HydranetManager(model_path=MagicMock())
        # Mock the property manually on the instance
        m._config_dict = {}
        type(m).configs = property(lambda self: self._config_dict, 
                                   lambda self, v: self._config_dict.update(v))
        return m

def test_translate_targets(manager):
    """Verify ln_ -> lr_ translation."""
    inputs = ["ln_sb_best", "ns_best", "lr_os_best"]
    expected = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
    assert manager._translate_targets(inputs) == expected

def test_augment_dataframe_unlogging(manager):
    """Verify ln_ column is unlogged into lr_ column."""
    # ln(100+1) is approx 4.61512
    df = pd.DataFrame({"ln_sb_best": [4.61512051681]})
    requested = ["lr_sb_best"]
    
    augmented = manager._augment_dataframe(df, requested)
    
    assert "lr_sb_best" in augmented.columns
    assert pytest.approx(augmented["lr_sb_best"].iloc[0], abs=1e-5) == 100.0

def test_augment_dataframe_binarization_from_raw(manager):
    """Verify _binarized is derived from lr_ column."""
    df = pd.DataFrame({"lr_sb_best": [0.0, 5.5, 0.0, 100.0]})
    requested = ["lr_sb_best_binarized"]
    
    augmented = manager._augment_dataframe(df, requested)
    
    assert "lr_sb_best_binarized" in augmented.columns
    expected = [0.0, 1.0, 0.0, 1.0]
    assert augmented["lr_sb_best_binarized"].tolist() == expected

def test_augment_dataframe_binarization_from_log(manager):
    """Verify _binarized is derived from ln_ column if lr_ is missing."""
    df = pd.DataFrame({"ln_sb_best": [0.0, 1.5, 0.0]})
    requested = ["ln_sb_best_binarized"]
    
    augmented = manager._augment_dataframe(df, requested)
    
    assert "ln_sb_best_binarized" in augmented.columns
    assert augmented["ln_sb_best_binarized"].tolist() == [0.0, 1.0, 0.0]

def test_augment_dataframe_no_override(manager):
    """Verify JIT doesn't overwrite existing columns."""
    df = pd.DataFrame({"lr_sb_best": [999.0]})
    requested = ["lr_sb_best"]
    
    augmented = manager._augment_dataframe(df, requested)
    # Value should remain 999, not be unlogged from a non-existent ln_ column
    assert augmented["lr_sb_best"].iloc[0] == 999.0
