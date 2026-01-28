import pytest
import pandas as pd
import numpy as np
from views_hydranet.utils.utils_hydranet_outputs import validate_contract_dataframes

def test_validate_contract_dataframes_happy_path():
    """Valid data should pass."""
    df = pd.DataFrame({
        "pred_lr_sb": [[1.0, 2.0], [3.0, 4.0]]
    }, index=pd.MultiIndex.from_tuples([(500, 1), (500, 2)], names=["month_id", "priogrid_gid"]))
    
    # Should not raise
    validate_contract_dataframes([df])

def test_validate_contract_dataframes_detects_nan():
    """NaN values should trigger ValueError."""
    df = pd.DataFrame({
        "pred_lr_sb": [[1.0, np.nan]]
    }, index=pd.MultiIndex.from_tuples([(500, 1)], names=["month_id", "priogrid_gid"]))
    
    with pytest.raises(ValueError, match="contains 1 non-finite values"):
        validate_contract_dataframes([df])

def test_validate_contract_dataframes_detects_inf():
    """Inf values should trigger ValueError."""
    df = pd.DataFrame({
        "pred_lr_sb": [[np.inf, 2.0]]
    }, index=pd.MultiIndex.from_tuples([(500, 1)], names=["month_id", "priogrid_gid"]))
    
    with pytest.raises(ValueError, match="contains 1 non-finite values"):
        validate_contract_dataframes([df])

def test_validate_contract_dataframes_detects_ocean():
    """priogrid_gid=0 should trigger ValueError."""
    df = pd.DataFrame({
        "pred_lr_sb": [[1.0]]
    }, index=pd.MultiIndex.from_tuples([(500, 0)], names=["month_id", "priogrid_gid"]))
    
    with pytest.raises(ValueError, match="contains ocean cells"):
        validate_contract_dataframes([df])

def test_validate_contract_dataframes_detects_empty():
    """Empty list or DF should trigger ValueError."""
    with pytest.raises(ValueError, match="list is empty"):
        validate_contract_dataframes([])
        
    with pytest.raises(ValueError, match="Sequence 0 is empty"):
        validate_contract_dataframes([pd.DataFrame()])
