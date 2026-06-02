"""
TDD tests for C-93: _evaluate_sweep and _load_model_artifact decomposition.

Tests the extraction of model loading from _setup_evaluation() and the new
_evaluate_sweep() override on HydranetManager.
"""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch

pytest.importorskip("views_pipeline_core")

from views_hydranet.manager.hydranet_manager import HydranetManager

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


def _mock_manager(tmp_path):
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


class TestGreenLoadModelArtifact:
    """Unit tests for the extracted _load_model_artifact() method."""

    def test_calls_fetcher(self, tmp_path):
        """_load_model_artifact() must delegate to ModelArtifactFetcher."""
        mgr = _mock_manager(tmp_path)
        mock_model = MagicMock(spec=torch.nn.Module)

        with (
            patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg,
            patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_maf,
        ):
            mock_cfg.return_value = _CFG
            mock_maf.return_value.fetch_model_artifact.return_value = (
                mock_model,
                "ts",
            )

            mgr._load_model_artifact()

            mock_maf.return_value.fetch_model_artifact.assert_called_once()

    def test_forwards_artifact_name(self, tmp_path):
        """artifact_name must reach fetch_model_artifact(model_artifact_name=...)."""
        mgr = _mock_manager(tmp_path)
        mock_model = MagicMock(spec=torch.nn.Module)

        with (
            patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg,
            patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_maf,
        ):
            mock_cfg.return_value = _CFG
            mock_maf.return_value.fetch_model_artifact.return_value = (
                mock_model,
                "ts",
            )

            mgr._load_model_artifact("my_specific_artifact")

            mock_maf.return_value.fetch_model_artifact.assert_called_once_with(
                model_artifact_name="my_specific_artifact"
            )

    def test_returns_model_not_tuple(self, tmp_path):
        """Must return the model object, not the (model, timestamp) tuple."""
        mgr = _mock_manager(tmp_path)
        mock_model = MagicMock(spec=torch.nn.Module)

        with (
            patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg,
            patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_maf,
        ):
            mock_cfg.return_value = _CFG
            mock_maf.return_value.fetch_model_artifact.return_value = (
                mock_model,
                "ts",
            )

            result = mgr._load_model_artifact()

            assert result is mock_model


class TestGreenSetupEvaluationRefactored:
    """Tests that _setup_evaluation accepts a model parameter and does not load one."""

    def test_uses_provided_model(self, tmp_path):
        """The model passed to _setup_evaluation must reach the InferenceOrchestrator."""
        mgr = _mock_manager(tmp_path)
        sentinel_model = MagicMock(spec=torch.nn.Module)

        with (
            patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg,
            patch.object(mgr, "_run_data_pipeline") as mock_pipeline,
            patch.object(mgr, "_run_preflight_check"),
            patch("views_hydranet.manager.hydranet_manager.ConfigInitializer") as mock_ci,
            patch("views_hydranet.manager.hydranet_manager.InferenceOrchestrator") as mock_orc_cls,
            patch("views_hydranet.manager.hydranet_manager.VisualDiagnostics"),
            patch("views_hydranet.manager.hydranet_manager.log_device_report"),
            patch(
                "views_hydranet.manager.hydranet_manager.get_rolling_origin_indices",
                return_value=[0],
            ),
        ):
            mock_cfg.return_value = _CFG
            mock_ci.return_value.get_config.return_value = _CFG
            mock_handler = MagicMock()
            mock_handler.shape = (10, 10, 1, 12)
            mock_pipeline.return_value = (mock_handler, MagicMock(), MagicMock())

            mgr._setup_evaluation("calibration", sentinel_model)

            mock_orc_cls.assert_called_once()
            _, call_kwargs = mock_orc_cls.call_args
            # model is the second positional arg to InferenceOrchestrator
            call_args = mock_orc_cls.call_args[0]
            assert call_args[1] is sentinel_model

    def test_does_not_load_model(self, tmp_path):
        """_setup_evaluation must never instantiate ModelArtifactFetcher."""
        mgr = _mock_manager(tmp_path)

        with (
            patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg,
            patch.object(mgr, "_run_data_pipeline") as mock_pipeline,
            patch.object(mgr, "_run_preflight_check"),
            patch("views_hydranet.manager.hydranet_manager.ConfigInitializer") as mock_ci,
            patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_maf,
            patch("views_hydranet.manager.hydranet_manager.InferenceOrchestrator"),
            patch("views_hydranet.manager.hydranet_manager.VisualDiagnostics"),
            patch("views_hydranet.manager.hydranet_manager.log_device_report"),
            patch(
                "views_hydranet.manager.hydranet_manager.get_rolling_origin_indices",
                return_value=[0],
            ),
        ):
            mock_cfg.return_value = _CFG
            mock_ci.return_value.get_config.return_value = _CFG
            mock_handler = MagicMock()
            mock_handler.shape = (10, 10, 1, 12)
            mock_pipeline.return_value = (mock_handler, MagicMock(), MagicMock())

            mgr._setup_evaluation("calibration", MagicMock(spec=torch.nn.Module))

            mock_maf.assert_not_called()


class TestGreenEvaluateSweep:
    """Tests for the new _evaluate_sweep() override."""

    def test_returns_dict(self, tmp_path):
        """_evaluate_sweep must return dict[str, list[PredictionFrame]]."""
        mgr = _mock_manager(tmp_path)
        mock_model = MagicMock(spec=torch.nn.Module)
        targets = ["lr_sb_best", "lr_ns_best", "lr_os_best"]

        mock_pf = MagicMock()
        pf_dict = {t: mock_pf for t in targets}
        mock_orchestrator = MagicMock()
        mock_orchestrator.generate_prediction_frames.return_value = [pf_dict]

        from types import SimpleNamespace

        mock_ctx = SimpleNamespace(
            handler=MagicMock(),
            scaler=MagicMock(),
            origins=[0],
            all_targets=targets,
            orchestrator=mock_orchestrator,
        )

        with patch.object(HydranetManager, "_setup_evaluation", return_value=mock_ctx):
            result = mgr._evaluate_sweep("calibration", mock_model)

        assert isinstance(result, dict)

    def test_uses_provided_model(self, tmp_path):
        """The model object passed to _evaluate_sweep must be the same one
        that reaches _setup_evaluation (identity check)."""
        mgr = _mock_manager(tmp_path)
        sentinel_model = MagicMock(spec=torch.nn.Module)
        targets = ["lr_sb_best"]

        mock_pf = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.generate_prediction_frames.return_value = [{t: mock_pf for t in targets}]

        from types import SimpleNamespace

        mock_ctx = SimpleNamespace(
            handler=MagicMock(),
            scaler=MagicMock(),
            origins=[0],
            all_targets=targets,
            orchestrator=mock_orchestrator,
        )

        with patch.object(
            HydranetManager, "_setup_evaluation", return_value=mock_ctx
        ) as mock_setup:
            mgr._evaluate_sweep("calibration", sentinel_model)

            mock_setup.assert_called_once_with("calibration", sentinel_model)

    def test_all_targets_present(self, tmp_path):
        """All target names must appear as keys in the returned dict."""
        mgr = _mock_manager(tmp_path)
        targets = ["lr_sb_best", "lr_ns_best", "lr_os_best", "by_sb_best"]

        mock_orchestrator = MagicMock()
        mock_orchestrator.generate_prediction_frames.return_value = [
            {t: MagicMock() for t in targets}
        ]

        from types import SimpleNamespace

        mock_ctx = SimpleNamespace(
            handler=MagicMock(),
            scaler=MagicMock(),
            origins=[0],
            all_targets=targets,
            orchestrator=mock_orchestrator,
        )

        with patch.object(HydranetManager, "_setup_evaluation", return_value=mock_ctx):
            result = mgr._evaluate_sweep("calibration", MagicMock())

        assert set(result.keys()) == set(targets)

    def test_calls_generate_prediction_frames(self, tmp_path):
        """The orchestrator's generate_prediction_frames must be called exactly once."""
        mgr = _mock_manager(tmp_path)
        targets = ["lr_sb_best"]

        mock_orchestrator = MagicMock()
        mock_orchestrator.generate_prediction_frames.return_value = [
            {t: MagicMock() for t in targets}
        ]

        from types import SimpleNamespace

        mock_ctx = SimpleNamespace(
            handler=MagicMock(),
            scaler=MagicMock(),
            origins=[0],
            all_targets=targets,
            orchestrator=mock_orchestrator,
        )

        with patch.object(HydranetManager, "_setup_evaluation", return_value=mock_ctx):
            mgr._evaluate_sweep("calibration", MagicMock())

        mock_orchestrator.generate_prediction_frames.assert_called_once()
