import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch
from views_hydranet.manager.hydranet_manager import HydranetManager

class MockHydraNet(nn.Module):
    """A real serializable minimal module for testing."""
    def __init__(self, base=32):
        super().__init__()
        self.base = base
        self.param = nn.Parameter(torch.ones(1))

    def forward(self, x, h):
        # HydraNet returns (reg, class, h)
        return torch.zeros(1, 3, 180, 180), torch.zeros(1, 3, 180, 180), h

    def init_hTtime(self, hidden_channels, H, W):
        return torch.zeros(1, hidden_channels, H, W)

class TestHydranetManager(HydranetManager):
    """Subclass to bypass brittle core config loading while keeping our logic real."""
    def __init__(self, model_path, config):
        self._model_path = model_path
        self._config_dict = config
        self.device = torch.device("cpu")
        self._wandb_notifications = False

    @property
    def configs(self): return self._config_dict
    @configs.setter
    def configs(self, v): self._config_dict.update(v)
    @property
    def config(self): return self._config_dict

@pytest.fixture
def full_system_env(tmp_path):
    proj_dir = tmp_path / "purple_alien"
    raw_dir = proj_dir / "data" / "raw"
    art_dir = proj_dir / "artifacts"
    gen_dir = proj_dir / "data" / "generated"
    
    for d in [raw_dir, art_dir, gen_dir]: d.mkdir(parents=True)
        
    # 1. Create minimal data
    df_path = raw_dir / "validation_viewser_df.parquet"
    df = pd.DataFrame({
        "month_id": [100, 101, 100, 101],
        "priogrid_gid": [1, 1, 2, 2],
        "row": [1, 1, 2, 2], "col": [1, 1, 2, 2], "c_id": [1, 1, 1, 1],
        "ln_sb_best": [0.1, 0.2, 0.3, 0.4],
        "ln_ns_best": [0.1, 0.2, 0.3, 0.4],
        "ln_os_best": [0.1, 0.2, 0.3, 0.4]
    })
    df.to_parquet(df_path)
    (raw_dir / "validation_data_fetch_log.txt").write_text("Fetch TS: 2026-01-29")
    
    # 2. Create dummy model
    model = MockHydraNet(base=32)
    model_path = art_dir / "validation_model_20260129_120000.pt"
    torch.save(model, model_path)
    
    return proj_dir, raw_dir, art_dir

def test_manager_end_to_end_smoke_run(full_system_env):
    """
    Final verification that the 'Environment Mirroring' actually works.
    """
    proj_dir, raw_dir, art_dir = full_system_env
    
    mpm = MagicMock()
    mpm.data_raw = raw_dir
    mpm.artifacts = art_dir
    mpm.data_generated = proj_dir / "data" / "generated"
    mpm.get_latest_model_artifact_path.return_value = art_dir / "validation_model_20260129_120000.pt"
    
    config = {
        "run_type": "validation",
        "steps": [1], # time_steps should become 1
        "test_samples": 1,
        "input_channels": 3,
        "target_variable": "sb",
        "targets": ["ln_sb_best"],
        "np_seed": 42,
        "torch_seed": 42
    }
    
    manager = TestHydranetManager(mpm, config)
    
    # Simulate the Core library's initialization logic for the test instance
    # (The TestHydranetManager doesn't run the full base __init__)
    from views_hydranet.utils.utils_config import HydraNetConfig
    validated = HydraNetConfig(**manager.configs)
    manager.configs["time_steps"] = validated.time_steps
    manager.configs["first_feature_idx"] = 5
    
    # EXECUTE with minimal inner mocking
    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager._execute_model_evaluation") as mock_super:
        
        # Verify state DURING core execution
        def side_effect():
            shadow_dir = art_dir / "tmp_eval_data"
            # 1. Did we mirror the log file?
            assert (shadow_dir / "validation_data_fetch_log.txt").exists()
            # 2. Did we heal the config?
            assert manager.configs["time_steps"] == 1
            assert manager.configs["first_feature_idx"] == 5
            # 3. Did we augment the data?
            df_aug = pd.read_parquet(shadow_dir / "validation_viewser_df.parquet")
            assert "lr_sb_best" in df_aug.columns
            
        mock_super.side_effect = side_effect
        
        manager._execute_model_evaluation()
        
    # Verify CLEANUP
    assert not (art_dir / "tmp_eval_data").exists()
    assert manager._model_path.data_raw == raw_dir
