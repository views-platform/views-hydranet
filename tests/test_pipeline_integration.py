from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
from views_hydranet.manager.hydranet_manager import HydranetManager

# Minimal valid config for the Boring Architecture (Updated for ADR 020)
TOY_CONFIG = {
    "run_type": "calibration",
    "steps": [1],
    "n_posterior_samples": 2,
    "total_lessons": 1,
    "windows_per_lesson": 1,
    "input_channels": 3,
    "output_channels": 1,
    "model": "HydraBNUNet06_LSTM4",
    "window_dim": 4,
    "height": 4,
    "width": 4,
    "total_hidden_channels": 32,
    "dropout_rate": 0.0,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "scheduler": "WarmupDecay",
    "warmup_steps": 1,
    "loss_reg": "lr_b",
    "loss_class": "lr_b",
    "loss_reg_a": 1,
    "loss_reg_c": 1,
    "loss_class_gamma": 1,
    "loss_class_alpha": 1,
    "freeze_h": "hl",
    "evaluation_mode": "stochastic",
    "aggregate_method": "geometric_mean",
    "np_seed": 4,
    "torch_seed": 4,
    "min_events": 0,
    "slope_ratio": 0.5,
    "roof_ratio": 0.5,
    "max_ratio": 0.95,
    "min_ratio": 0.05,
    "time_steps": 1,
    # Ledger Roles
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "index_names": ["month_id", "priogrid_gid"],
    "spatial_cols": ["row", "col"],
    "row_offset": 0,
    "col_offset": 0,
    "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "identity_cols": ["month_id", "priogrid_gid", "row", "col"],
    # Outbound / Evaluation (ADR 032 Alignment)
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    # ADR 046 Compliance
    "transformations": {"log1p": ["lr_sb_best", "lr_ns_best", "lr_os_best"], "identity": []},
    "derivations": {
        "binary": [
            {"from": "lr_sb_best", "to": "by_sb_best", "threshold": 0},
            {"from": "lr_ns_best", "to": "by_ns_best", "threshold": 0},
            {"from": "lr_os_best", "to": "by_os_best", "threshold": 0},
        ]
    },
}


@pytest.fixture
def toy_dataframe():
    """Generates a perfectly aligned 4x4 grid over 10 months."""
    rows = []
    for t in range(100, 110):
        for r in range(4):
            for c in range(4):
                rows.append(
                    {
                        "month_id": t,
                        "priogrid_gid": r * 4 + c + 1,
                        "row": r,
                        "col": c,
                        "lr_sb_best": 0.5,
                        "lr_ns_best": 0.1,
                        "lr_os_best": 0.0,
                    }
                )
    return pd.DataFrame(rows)


class TestPipelineIntegration:
    """
    Verifies the End-to-End flow:
    DataFrame -> Custodian -> Inference -> Wrapper (The Fix) -> Reconstruction
    """

    @patch("views_hydranet.manager.hydranet_manager.FeatureScaler")
    @patch("views_hydranet.manager.hydranet_manager.DataFetcher")
    @patch("views_hydranet.manager.hydranet_manager.ConfigInitializer")
    @patch(
        "views_pipeline_core.managers.model.model.ForecastingModelManager.__init__",
        return_value=None,
    )
    def test_evaluate_model_artifact_flow(
        self,
        mock_fmm_init,
        mock_config_init,
        mock_fetcher_cls,
        mock_scaler_cls,
        toy_dataframe,
        tmp_path,
    ):
        # 1. Setup Environment
        mock_fetcher_instance = mock_fetcher_cls.return_value
        mock_fetcher_instance.fetch_df.return_value = toy_dataframe
        # Standardize just returns the df
        mock_fetcher_cls.standardize_raw_df.side_effect = lambda df, cfg: df

        # Config Handshake bypass
        mock_config_init.return_value.get_config.return_value = TOY_CONFIG

        # Scaler Mock
        mock_scaler = mock_scaler_cls.return_value
        # Matches regression outputs
        mock_scaler.configured_columns = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
        mock_scaler.fit_transform.side_effect = lambda df: df
        mock_scaler.inverse_transform.side_effect = lambda df: df
        # NEW: Support volume inversion in the mock
        mock_scaler.inverse_transform_volume.side_effect = lambda vh: vh

        # Mock ModelPathManager
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path

        # 2. Instantiate Manager
        manager = HydranetManager(model_path=mpm)
        manager.device = torch.device("cpu")

        # FIX: Mock the configs property using PropertyMock since we mocked __init__
        with patch(
            "views_pipeline_core.managers.model.model.ForecastingModelManager.configs",
            new_callable=PropertyMock,
        ) as mock_configs_prop:
            mock_configs_prop.return_value = TOY_CONFIG

            real_model = HydraBNUNet06_LSTM4(
                input_channels=3, total_hidden_channels=32, output_channels=1, dropout_rate=0.0
            )

            with (
                patch(
                    "views_hydranet.manager.hydranet_manager.ModelArtifactFetcher"
                ) as mock_art_fetch_cls,
                patch(
                    "views_hydranet.manager.hydranet_manager.InferenceOrchestrator"
                ) as mock_eval_cls,
            ):
                mock_art_fetch_cls.return_value.fetch_model_artifact.return_value = (
                    real_model,
                    "toy_artifact",
                )

                # Mock Orchestrator output — all 6 prediction columns required by _to_pf_dict
                df_mock = toy_dataframe.copy()
                df_mock["pred_lr_sb_best"] = 0.5
                df_mock["pred_lr_ns_best"] = 0.5
                df_mock["pred_lr_os_best"] = 0.5
                df_mock["pred_by_sb_best"] = 0.9
                df_mock["pred_by_ns_best"] = 0.9
                df_mock["pred_by_os_best"] = 0.9
                df_mock = df_mock.set_index(["month_id", "priogrid_gid"])
                mock_eval_cls.return_value.generate_forecasts.return_value = [df_mock]

                # 4. EXECUTE THE CRITICAL PATH
                predictions = manager._evaluate_model_artifact(eval_type="calibration")

                # 5. Verify Output — dict[str, list[PredictionFrame]]
                assert isinstance(predictions, dict)
                assert len(predictions) == 6  # 3 reg + 3 cls targets

                # Verify all target keys present
                assert "lr_sb_best" in predictions
                assert "by_sb_best" in predictions
                assert "lr_ns_best" in predictions

                # Verify each value is a non-empty list of PredictionFrames
                pf_sb = predictions["lr_sb_best"][0]
                assert pf_sb.y_pred.ndim == 2
                assert len(pf_sb.y_pred) > 0

                # Verify identifiers preserved
                assert "time" in pf_sb.identifiers
                assert "unit" in pf_sb.identifiers

    @patch("views_hydranet.manager.hydranet_manager.FeatureScaler")
    @patch("views_hydranet.manager.hydranet_manager.DataFetcher")
    @patch("views_hydranet.manager.hydranet_manager.ConfigInitializer")
    @patch(
        "views_pipeline_core.managers.model.model.ForecastingModelManager.__init__",
        return_value=None,
    )
    def test_forecast_model_artifact_flow(
        self,
        mock_fmm_init,
        mock_config_init,
        mock_fetcher_cls,
        mock_scaler_cls,
        toy_dataframe,
        tmp_path,
    ):
        # 1. Setup Environment
        mock_fetcher_instance = mock_fetcher_cls.return_value
        mock_fetcher_instance.fetch_df.return_value = toy_dataframe
        mock_fetcher_cls.standardize_raw_df.side_effect = lambda df, cfg: df
        mock_config_init.return_value.get_config.return_value = TOY_CONFIG

        mock_scaler = mock_scaler_cls.return_value
        mock_scaler.configured_columns = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
        mock_scaler.fit_transform.side_effect = lambda df: df
        mock_scaler.inverse_transform.side_effect = lambda df: df
        mock_scaler.inverse_transform_volume.side_effect = lambda vh: vh

        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path

        manager = HydranetManager(model_path=mpm)
        manager.device = torch.device("cpu")

        with patch(
            "views_pipeline_core.managers.model.model.ForecastingModelManager.configs",
            new_callable=PropertyMock,
        ) as mock_configs_prop:
            mock_configs_prop.return_value = TOY_CONFIG

            real_model = HydraBNUNet06_LSTM4(3, 32, 1, 0.0)
            with (
                patch(
                    "views_hydranet.manager.hydranet_manager.ModelArtifactFetcher"
                ) as mock_art_fetch_cls,
                patch(
                    "views_hydranet.utils.inference_orchestrator.HydraNetInference"
                ) as mock_inf_cls,
            ):
                mock_art_fetch_cls.return_value.fetch_model_artifact.return_value = (
                    real_model,
                    "toy_artifact",
                )

                # Mock Inference Result
                # TOY_CONFIG has 3 reg target and 3 cls targets.
                # Total 6 signal channels.
                # Shape must be [T, H, W, C, S] for numpy input to wrap_predictions
                posterior = np.zeros((1, 4, 4, 6, 2))
                mock_inf_cls.return_value.generate_posterior_samples.return_value = (
                    posterior,
                    None,
                )

                # 4. EXECUTE FORECAST PATH
                forecasts = manager._forecast_model_artifact()

                # 5. Verify Output — dict[str, PredictionFrame]
                assert isinstance(forecasts, dict)
                assert len(forecasts) == 6  # 3 reg + 3 cls targets

                assert "lr_sb_best" in forecasts
                assert "by_sb_best" in forecasts

                pf_sb = forecasts["lr_sb_best"]
                assert pf_sb.y_pred.ndim == 2
                assert len(pf_sb.y_pred) > 0
                assert "time" in pf_sb.identifiers
                assert "unit" in pf_sb.identifiers
