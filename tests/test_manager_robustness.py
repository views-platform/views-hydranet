from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from views_hydranet.manager.hydranet_manager import HydranetManager


@pytest.fixture
def manager_robust_env(tmp_path):
    """Real filesystem environment for robustness testing."""
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    gen_dir = tmp_path / "generated"
    gen_dir.mkdir()

    with patch("views_hydranet.manager.hydranet_manager.HydranetManager.__init__", return_value=None):
        with patch("views_hydranet.manager.hydranet_manager.setup_device", return_value="cpu"):
            m = HydranetManager(model_path=MagicMock())
            m._model_path = MagicMock()
            m._model_path.artifacts = art_dir
            m._model_path.data_generated = gen_dir
            m.device = "cpu"

            m._hydranet_config = {
                "run_type": "validation",
                "time_steps": 3,
                "test_samples": 1,
                "input_channels": 3,
                "target_variable": "sb",
                "targets": ["lr_sb_best"],
                "log1p": ["lr_sb_best"],
                "asinh": [],
                "identity": []
            }
            # Attach a mock _load_model_artifact to the instance to allow patching
            m._load_model_artifact = MagicMock(return_value=(MagicMock(), "ts"))
            return m

def test_multitask_merging_alignment(manager_robust_env):
    """
    PROVE the fix for the Flattening Trap using real method logic.
    """
    manager = manager_robust_env
    manager._hydranet_config["targets"] = ["lr_sb_best", "lr_ns_best"]
    manager._hydranet_config["log1p"] = ["lr_sb_best", "lr_ns_best"]
    manager._hydranet_config["target_variable"] = "" # Both

    with patch.object(manager, "_load_model_artifact", return_value=(MagicMock(), "ts")):
        with patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetcher_cls, \
             patch("views_hydranet.manager.hydranet_manager.df_to_vol") as mock_converter:
            with patch("views_hydranet.manager.hydranet_manager.HydraNetInference") as mock_inf_cls:
                with patch("views_hydranet.manager.hydranet_manager.zstack_to_contract_df") as mock_conv:
                    with patch("views_hydranet.manager.hydranet_manager.pickle.dump"):
                        with patch("views_hydranet.manager.hydranet_manager.validate_contract_dataframes"):

                            mock_fetcher = mock_fetcher_cls.return_value
                            # USE REAL DATAFRAME
                            real_df = pd.DataFrame(columns=["lr_sb_best", "lr_ns_best", "lr_os_best"])
                            mock_fetcher.fetch.return_value = real_df

                            mock_converter.return_value = np.zeros((10, 10, 10, 8))
                            mock_inference = mock_inf_cls.return_value
                            mock_inference.generate_posterior_samples.return_value = (MagicMock(), MagicMock())

                            def side_effect(posterior_zstack, meta_zstack, target, **kwargs):
                                return [pd.DataFrame({target: [0.5]}, index=pd.MultiIndex.from_tuples([(1,1)], names=["month_id", "priogrid_gid"]))]
                            mock_conv.side_effect = side_effect

                            results = HydranetManager._evaluate_model_artifact(manager, eval_type="offline")

                            assert len(results) == 7
                            assert "pred_lr_sb_best" in results[0].columns
                            assert "pred_lr_ns_best" in results[0].columns

def test_partition_aware_windows(manager_robust_env):
    """Verify that forecasting partition correctly defaults to 1 window."""
    manager = manager_robust_env
    manager._hydranet_config["run_type"] = "forecasting"

    with patch.object(manager, "_load_model_artifact", return_value=(MagicMock(), "ts")):
        with patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetcher_cls, \
             patch("views_hydranet.manager.hydranet_manager.df_to_vol") as mock_converter:
            with patch("views_hydranet.manager.hydranet_manager.HydraNetInference") as mock_inf_cls:
                with patch("views_hydranet.manager.hydranet_manager.zstack_to_contract_df") as mock_conv:
                    with patch("views_hydranet.manager.hydranet_manager.pickle.dump"):
                        with patch("views_hydranet.manager.hydranet_manager.validate_contract_dataframes"):

                            mock_fetcher = mock_fetcher_cls.return_value
                            # USE REAL DATAFRAME
                            real_df = pd.DataFrame(columns=["lr_sb_best"])
                            mock_fetcher.fetch.return_value = real_df

                            mock_converter.return_value = np.zeros((100, 10, 10, 8))
                            mock_inference = mock_inf_cls.return_value
                            mock_inference.generate_posterior_samples.return_value = (MagicMock(), MagicMock())
                            mock_conv.return_value = [pd.DataFrame({"lr_sb_best": [0.5]})]

                            results = HydranetManager._evaluate_model_artifact(manager, eval_type="offline")
                            assert len(results) == 1
