from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from views_hydranet.manager.hydranet_manager import HydranetManager


@pytest.fixture
def manager_env(tmp_path, valid_config_dict):
    raw_dir = tmp_path / "data" / "raw"
    art_dir = tmp_path / "artifacts"
    raw_dir.mkdir(parents=True)
    art_dir.mkdir(parents=True)
    pd.DataFrame({"lr_sb_best": [1.0]}).to_parquet(raw_dir / "validation_viewser_df.parquet")
    (raw_dir / "validation_data_fetch_log.txt").write_text("Fetched")

    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None), \
         patch("views_hydranet.manager.hydranet_manager.setup_device", return_value="cpu"):

        m = HydranetManager(model_path=MagicMock())
        m._model_path = MagicMock()
        m._model_path.data_raw = raw_dir
        m._model_path.artifacts = art_dir
        m._hydranet_config = valid_config_dict
        return m, raw_dir, art_dir

def test_manager_evaluation_lifecycle_explicit_augmentation(manager_env):
    manager, raw_dir, art_dir = manager_env
    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager._execute_model_evaluation"):
        manager._execute_model_evaluation()
    assert not (art_dir / "tmp_eval_data").exists()

def test_manager_restoration_under_chaos(manager_env):
    manager, raw_dir, _ = manager_env
    with patch("views_pipeline_core.managers.model.model.ForecastingModelManager._execute_model_evaluation", side_effect=RuntimeError):
        with pytest.raises(RuntimeError):
            manager._execute_model_evaluation()
    assert manager._model_path.data_raw == raw_dir
