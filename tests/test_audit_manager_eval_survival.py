from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("views_pipeline_core")

from views_pipeline_core.data.prediction_frame import PredictionFrame

from views_hydranet.manager.hydranet_manager import HydranetManager


def _pf_dict(values: dict[str, float], n: int, month_id: int = 124) -> dict:
    """Build a mock list[dict[str, PredictionFrame]] for generate_prediction_frames."""
    time_arr = np.array([month_id] * n, dtype=np.int32)
    unit_arr = np.array(range(1, n + 1), dtype=np.int32)
    return {
        target: PredictionFrame(
            y_pred=np.full((n, 1), val),
            identifiers={"time": time_arr, "unit": unit_arr},
        )
        for target, val in values.items()
    }


# AUDIT CONFIG: Point mode, arithmetic mean, heterogeneous scales
AUDIT_CFG = {
    "run_type": "calibration",
    "steps": [1],
    "time_steps": 1,
    "input_channels": 3,
    "output_channels": 1,
    "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    "identity_cols": ["month_id", "priogrid_gid"],
    "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "transformations": {
        "log1p": ["lr_sb_best"],
        "asinh": ["lr_ns_best"],
        "identity": ["lr_os_best"],
    },
    "derivations": {
        "binary": [
            {"from": "lr_sb_best", "to": "by_sb_best", "threshold": 0},
            {"from": "lr_ns_best", "to": "by_ns_best", "threshold": 0},
            {"from": "lr_os_best", "to": "by_os_best", "threshold": 0},
        ]
    },
    "height": 4,
    "width": 4,
    "index_names": ["month_id", "priogrid_gid"],
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "row_offset": 0,
    "col_offset": 0,
    "model": "Dummy",
    "window_dim": 2,
    "total_hidden_channels": 8,
    "dropout_rate": 0.0,
    "weight_init": "norm",
    "h_init": "zero",
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "windows_per_lesson": 1,
    "scheduler": "none",
    "warmup_steps": 1,
    "clip_grad_norm": True,
    "loss_reg": "mse",
    "loss_class": "bce",
    "loss_reg_a": 1,
    "loss_reg_c": 1,
    "loss_class_gamma": 1,
    "loss_class_alpha": 1,
    "total_lessons": 1,
    "n_posterior_samples": 10,
    "np_seed": 1,
    "torch_seed": 1,
    "min_events": 0,
    "slope_ratio": 0.1,
    "roof_ratio": 0.1,
    "max_ratio": 0.9,
    "min_ratio": 0.1,
    "freeze_h": "none",
    "sampling_strategy": "threshold",
    "evaluation_mode": "point",
    "aggregate_method": "arithmetic_mean",
}


class TestRedManagerEvalHardAudit:
    """8 Hard Gates to falsify the Survival Sequence orchestration."""

    def test_gates_1_to_8_eval_survival(self, tmp_path):
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path

        # 1. Setup History
        df_hist = pd.DataFrame(
            {
                "month_id": sorted(list(range(100, 124)) * 16),
                "priogrid_gid": list(range(1, 17)) * 24,
                "row": [0] * 384,
                "col": [0] * 384,
                "lr_sb_best": [10.0] * 384,
                "lr_ns_best": [10.0] * 384,
                "lr_os_best": [10.0] * 384,
            }
        )

        with patch(
            "views_pipeline_core.managers.model.model.ForecastingModelManager.__init__",
            return_value=None,
        ):
            manager = HydranetManager(model_path=mpm)
            manager.device = torch.device("cpu")
            manager._config_manager = MagicMock()
            manager._wandb_notifications = False
            manager._use_prediction_store = False

            with patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg:
                mock_cfg.return_value = AUDIT_CFG

                with (
                    patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch_cls,
                    patch(
                        "views_hydranet.manager.hydranet_manager.ModelArtifactFetcher"
                    ) as mock_art_fetch_cls,
                    patch(
                        "views_hydranet.manager.hydranet_manager.InferenceOrchestrator"
                    ) as mock_eval_cls,
                ):
                    # 1. Mock DataFetcher
                    mock_fetch_cls.return_value.fetch_df.return_value = df_hist
                    mock_fetch_cls.standardize_raw_df.side_effect = lambda x, y: x

                    # 2. Mock ModelFetcher
                    mock_art_fetch_cls.return_value.fetch_model_artifact.return_value = (
                        MagicMock(),
                        "audit",
                    )

                    # 3. Mock Evaluator — generate_prediction_frames returns
                    # list[dict[str, PredictionFrame]] (pandas-free path)
                    mock_eval_cls.return_value.generate_prediction_frames.return_value = [
                        _pf_dict(
                            {
                                "lr_sb_best": 100.0,
                                "lr_ns_best": 100.0,
                                "lr_os_best": 100.0,
                                "by_sb_best": 0.9,
                                "by_ns_best": 0.9,
                                "by_os_best": 0.9,
                            },
                            n=16,
                        )
                    ]

                    # RUN EVALUATION
                    results = manager._evaluate_model_artifact(eval_type="audit")
                    pf_sb = results["lr_sb_best"][0]
                    pf_by_sb = results["by_sb_best"][0]
                    pf_ns = results["lr_ns_best"][0]

                    # GATES
                    np.testing.assert_allclose(pf_sb.y_pred[0, 0], 100.0, rtol=1e-5)
                    assert isinstance(pf_sb.y_pred[0, 0], (float, np.floating))
                    assert "lr_sb_best" in results
                    assert "by_sb_best" in results
                    assert pf_sb.y_pred.ndim == 2
                    np.testing.assert_allclose(pf_by_sb.y_pred[0, 0], 0.9, rtol=1e-5)
                    assert pf_sb.y_pred.ndim == 2  # Dense array, no list-in-cell
                    np.testing.assert_allclose(pf_ns.y_pred[0, 0], 100.0, rtol=1e-5)

    def test_gate_8_nuke_proof_heterogeneous(self, tmp_path):
        """Hard Gate 8: Verify multiple inverse functions in one volume."""
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path

        df_hist = pd.DataFrame(
            {
                "month_id": sorted(list(range(100, 124)) * 16),
                "priogrid_gid": list(range(1, 17)) * 24,
                "row": [0] * 384,
                "col": [0] * 384,
                "lr_sb_best": [1.0] * 384,
                "lr_ns_best": [1.0] * 384,
                "lr_os_best": [1.0] * 384,
            }
        )

        with patch(
            "views_pipeline_core.managers.model.model.ForecastingModelManager.__init__",
            return_value=None,
        ):
            manager = HydranetManager(model_path=mpm)
            manager.device = torch.device("cpu")
            manager._config_manager = MagicMock()
            manager._wandb_notifications = False
            manager._use_prediction_store = False
            with patch.object(
                HydranetManager,
                "configs",
                new_callable=PropertyMock,
            ) as mock_cfg:
                mock_cfg.return_value = AUDIT_CFG
                with (
                    patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch_cls,
                    patch(
                        "views_hydranet.manager.hydranet_manager.ModelArtifactFetcher"
                    ) as mock_art_fetch_cls,
                    patch(
                        "views_hydranet.manager.hydranet_manager.InferenceOrchestrator"
                    ) as mock_eval_cls,
                ):
                    mock_fetch_cls.return_value.fetch_df.return_value = df_hist
                    mock_fetch_cls.standardize_raw_df.side_effect = lambda x, y: x
                    mock_art_fetch_cls.return_value.fetch_model_artifact.return_value = (
                        MagicMock(),
                        "audit",
                    )

                    mock_eval_cls.return_value.generate_prediction_frames.return_value = [
                        _pf_dict(
                            {
                                "lr_sb_best": 10.0,
                                "lr_ns_best": 10.0,
                                "lr_os_best": 10.0,
                                "by_sb_best": 0.5,
                                "by_ns_best": 0.5,
                                "by_os_best": 0.5,
                            },
                            n=16,
                        )
                    ]

                    results = manager._evaluate_model_artifact(eval_type="audit")
                    np.testing.assert_allclose(
                        results["lr_sb_best"][0].y_pred[0, 0],
                        10.0,
                        rtol=1e-5,
                    )
                    np.testing.assert_allclose(
                        results["lr_ns_best"][0].y_pred[0, 0],
                        10.0,
                        rtol=1e-5,
                    )

    def test_gates_9_to_16_forecast_survival(self, tmp_path):
        """Hard Gates 9-16: Falsify the Survival Sequence in the Forecasting path."""
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path

        df_hist = pd.DataFrame(
            {
                "month_id": sorted(list(range(100, 124)) * 16),
                "priogrid_gid": list(range(1, 17)) * 24,
                "row": [0] * 384,
                "col": [0] * 384,
                "lr_sb_best": [10.0] * 384,
                "lr_ns_best": [10.0] * 384,
                "lr_os_best": [10.0] * 384,
            }
        )

        with patch(
            "views_pipeline_core.managers.model.model.ForecastingModelManager.__init__",
            return_value=None,
        ):
            manager = HydranetManager(model_path=mpm)
            manager.device = torch.device("cpu")
            manager._config_manager = MagicMock()
            manager._wandb_notifications = False
            manager._use_prediction_store = False

            with patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg:
                mock_cfg.return_value = AUDIT_CFG

                with (
                    patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch_cls,
                    patch(
                        "views_hydranet.manager.hydranet_manager.ModelArtifactFetcher"
                    ) as mock_art_fetch_cls,
                    patch(
                        "views_hydranet.utils.inference_orchestrator.HydraNetInference"
                    ) as mock_inf_cls,
                    patch("views_hydranet.manager.hydranet_manager.DataSniffer"),
                ):
                    # 5. EXECUTE (ADR 038 Unified Flow)
                    mock_fetch_cls.return_value.fetch_df.return_value = df_hist
                    mock_fetch_cls.standardize_raw_df.side_effect = lambda x, y: x

                    # Ensure mock model satisfies isinstance(model, Module)
                    mock_model = MagicMock(spec=torch.nn.Module)
                    mock_art_fetch_cls.return_value.fetch_model_artifact.return_value = (
                        mock_model,
                        "audit",
                    )

                    # posterior: (T, H, W, C) -> 3 targets * 2 heads = 6 channels
                    posterior = np.zeros((1, 4, 4, 6))
                    posterior[:, :, :, 0] = np.log1p(50.0)
                    mock_inf_cls.return_value.generate_posterior_samples.return_value = (
                        posterior,
                        None,
                    )

                    # RUN FORECAST
                    results = manager._forecast_model_artifact()
                    pf_sb = results["lr_sb_best"]

                    # GATES
                    np.testing.assert_allclose(pf_sb.y_pred[0, 0], 50.0, rtol=1e-5)
                    assert isinstance(pf_sb.y_pred[0, 0], (float, np.floating))
                    assert pf_sb.identifiers["time"][0] == 124


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
