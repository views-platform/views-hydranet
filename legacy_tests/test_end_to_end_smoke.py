from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
import torch.nn as nn

from views_hydranet.manager.hydranet_manager import HydranetManager


class MockHydraNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.base = base
        self.param = nn.Parameter(torch.ones(1))
    def forward(self, x, h):
        return torch.zeros(1, 3, 180, 180), torch.zeros(1, 3, 180, 180), h
    def init_hTtime(self, hidden_channels, H, W):
        return torch.zeros(1, hidden_channels, H, W)

@pytest.fixture
def full_system_env(tmp_path):
    proj_dir = tmp_path / "purple_alien"
    raw_dir = proj_dir / "data" / "raw"
    art_dir = proj_dir / "artifacts"
    gen_dir = proj_dir / "data" / "generated"
    for d in [raw_dir, art_dir, gen_dir]: d.mkdir(parents=True)
    df_path = raw_dir / "validation_viewser_df.parquet"
    pd.DataFrame({"month_id": [100, 101], "priogrid_gid": [1, 1], "lr_sb_best": [0.1, 0.2]}).to_parquet(df_path)
    (raw_dir / "validation_data_fetch_log.txt").write_text("Fetch TS: 2026-01-29")
    model = MockHydraNet(base=32)
    torch.save(model, art_dir / "model.pt")
    return proj_dir, raw_dir, art_dir

def test_manager_end_to_end_smoke_run(full_system_env, valid_config_dict):
    proj_dir, raw_dir, art_dir = full_system_env
    mpm = MagicMock()
    mpm.data_raw = raw_dir
    mpm.artifacts = art_dir
    mpm.get_latest_model_artifact_path.return_value = art_dir / "model.pt"

    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None):
        with patch("views_hydranet.manager.hydranet_manager.setup_device", return_value=torch.device("cpu")):
            manager = HydranetManager(model_path=mpm)
            manager._hydranet_config = valid_config_dict

            with patch("views_pipeline_core.managers.model.model.ForecastingModelManager._execute_model_evaluation"):
                manager._execute_model_evaluation()
                assert manager.config["time_steps"] == 36
