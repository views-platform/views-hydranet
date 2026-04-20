"""
Runtime memory hygiene tests for HydranetManager.

Verifies that _run_data_pipeline:
  1. Calls gc.collect() after consuming the DataFrame.
  2. Is used by all 3 artifact methods (train, evaluate, forecast).
"""

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("views_pipeline_core")

from views_hydranet.manager.hydranet_manager import HydranetManager

# Minimal config sufficient for the manager paths under test
_CFG = {
    "run_type": "calibration",
    "model": "HydraBNUNet06_LSTM4",
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    "features": ["feat_a"],
    "steps": list(range(1, 37)),
    "sweep": False,
    "diagnostic_visualizations": False,
    "evaluation_mode": "point",
    "aggregate_method": "arithmetic_mean",
    "np_seed": 42,
    "torch_seed": 42,
}

_ROWS = 100
_DF = pd.DataFrame(
    {
        "month_id": np.tile(np.arange(1, 11), 10),
        "priogrid_gid": np.repeat(np.arange(1, 11), 10),
        "row": np.repeat(np.arange(1, 11), 10),
        "col": np.ones(_ROWS, dtype=int),
        "c_id": np.ones(_ROWS, dtype=int),
        "lr_sb_best": np.random.rand(_ROWS),
        "lr_ns_best": np.random.rand(_ROWS),
        "lr_os_best": np.random.rand(_ROWS),
        "by_sb_best": np.random.rand(_ROWS),
        "by_ns_best": np.random.rand(_ROWS),
        "by_os_best": np.random.rand(_ROWS),
        "feat_a": np.random.rand(_ROWS),
    }
)


def _mock_manager(tmp_path):
    """Create a minimal mocked HydranetManager instance."""
    with patch.object(HydranetManager, "__init__", lambda self, *a, **kw: None):
        mgr = HydranetManager.__new__(HydranetManager)
        mgr.device = torch.device("cpu")
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path / "artifacts"
        mpm.artifacts.mkdir(exist_ok=True)
        mgr._model_path = mpm
        mgr.run_timestamp = "test_ts"
        return mgr


class TestRunDataPipelineCallsGcCollect:
    """Runtime test: _run_data_pipeline must call gc.collect()."""

    def test_gc_collect_called(self, tmp_path):
        mgr = _mock_manager(tmp_path)

        with (
            patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg,
            patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch_cls,
            patch("views_hydranet.manager.hydranet_manager.FeatureScaler") as mock_scaler_cls,
            patch("views_hydranet.manager.hydranet_manager.DataSniffer"),
            patch("views_hydranet.manager.hydranet_manager.VolumeHandler") as mock_vh_cls,
            patch("views_hydranet.manager.hydranet_manager.gc") as mock_gc,
        ):
            mock_cfg.return_value = _CFG
            mock_fetch_cls.return_value.fetch_df.return_value = _DF.copy()
            mock_fetch_cls.standardize_raw_df.return_value = _DF.copy()
            mock_scaler_cls.return_value.fit_transform.return_value = _DF.copy()
            mock_handler = MagicMock()
            mock_handler.shape = (10, 10, 1, 12)
            mock_vh_cls.from_df.return_value = mock_handler

            viz = MagicMock()
            mgr._run_data_pipeline(viz)

            assert mock_gc.collect.called, (
                "_run_data_pipeline() must call gc.collect() to release "
                "the DataFrame memory after volume construction."
            )


class TestArtifactMethodsDelegateToSharedPipeline:
    """Runtime tests: all 3 artifact methods must call _run_data_pipeline."""

    def test_train_delegates(self, tmp_path):
        mgr = _mock_manager(tmp_path)
        with (
            patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg,
            patch.object(mgr, "_run_data_pipeline") as mock_pipeline,
            patch.object(mgr, "_run_preflight_check"),
            patch("views_hydranet.manager.hydranet_manager.ConfigInitializer") as mock_ci,
            patch(
                "views_hydranet.manager.hydranet_manager.train_model_artifact",
                return_value=(MagicMock(), {}),
            ),
            patch("views_hydranet.manager.hydranet_manager.VisualDiagnostics"),
            patch("views_hydranet.manager.hydranet_manager.log_device_report"),
            patch("views_hydranet.manager.hydranet_manager.log_training_summary"),
        ):
            mock_cfg.return_value = _CFG
            mock_ci.return_value.get_config.return_value = _CFG
            mock_pipeline.return_value = (MagicMock(), MagicMock(), MagicMock())

            mgr._train_model_artifact()
            mock_pipeline.assert_called_once()

    def test_setup_evaluation_delegates(self, tmp_path):
        mgr = _mock_manager(tmp_path)
        with (
            patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg,
            patch.object(mgr, "_run_data_pipeline") as mock_pipeline,
            patch.object(mgr, "_run_preflight_check"),
            patch("views_hydranet.manager.hydranet_manager.ConfigInitializer") as mock_ci,
            patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_maf,
            patch("views_hydranet.manager.hydranet_manager.VisualDiagnostics"),
            patch("views_hydranet.manager.hydranet_manager.log_device_report"),
            patch(
                "views_hydranet.manager.hydranet_manager.get_rolling_origin_indices",
                return_value=[0],
            ),
        ):
            mock_cfg.return_value = _CFG
            mock_ci.return_value.get_config.return_value = _CFG
            mock_maf.return_value.fetch_model_artifact.return_value = (MagicMock(), "artifact")
            mock_handler = MagicMock()
            mock_handler.shape = (10, 10, 1, 12)
            mock_pipeline.return_value = (mock_handler, MagicMock(), MagicMock())

            mgr._setup_evaluation("calibration")
            mock_pipeline.assert_called_once()

    def test_forecast_delegates(self, tmp_path):
        mgr = _mock_manager(tmp_path)
        with (
            patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg,
            patch.object(mgr, "_run_data_pipeline") as mock_pipeline,
            patch.object(mgr, "_run_preflight_check"),
            patch("views_hydranet.manager.hydranet_manager.ConfigInitializer") as mock_ci,
            patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_maf,
            patch("views_hydranet.manager.hydranet_manager.InferenceOrchestrator") as mock_orc_cls,
            patch("views_hydranet.manager.hydranet_manager.VisualDiagnostics"),
            patch("views_hydranet.manager.hydranet_manager.log_device_report"),
        ):
            mock_cfg.return_value = _CFG
            mock_ci.return_value.get_config.return_value = _CFG
            mock_maf.return_value.fetch_model_artifact.return_value = (MagicMock(), "artifact")
            mock_handler = MagicMock()
            mock_handler.shape = (10, 10, 1, 12)
            mock_pipeline.return_value = (mock_handler, MagicMock(), MagicMock())
            mock_orc_cls.return_value.generate_prediction_frames.return_value = [
                {
                    t: MagicMock()
                    for t in (_CFG["regression_targets"] + _CFG["classification_targets"])
                }
            ]

            mgr._forecast_model_artifact()
            mock_pipeline.assert_called_once()
