"""
Falsification tests for claim:
"This model can run end-to-end both for backtesting/evaluation and for true future forecasts."

Audit date: 2026-04-19
Source: /falsify skill

Originally RED stubs (P-02, P-03, P-04). Converted to GREEN after fixes.
"""

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

pytest.importorskip("views_pipeline_core")

from views_hydranet.manager.hydranet_manager import HydranetManager
from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

MINIMAL_CFG = {
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
        "identity": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
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
    "n_posterior_samples": 2,
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


def _make_manager(tmp_path):
    """Create a minimal mocked HydranetManager."""
    mpm = MagicMock()
    mpm.data_raw = tmp_path
    mpm.artifacts = tmp_path

    import torch

    with patch(
        "views_pipeline_core.managers.model.model.ForecastingModelManager.__init__",
        return_value=None,
    ):
        manager = HydranetManager(model_path=mpm)
        manager.device = torch.device("cpu")
        manager._config_manager = MagicMock()
        manager._wandb_notifications = False
        manager._use_prediction_store = False
        manager.run_timestamp = "test"

    return manager


class TestArtifactNameSilentlyIgnored:
    """P-02 (C-56): artifact_name must reach fetch_model_artifact()."""

    def test_artifact_name_forwarded_to_fetcher(self, tmp_path):
        """artifact_name parameter must reach fetch_model_artifact()."""

        manager = _make_manager(tmp_path)

        with (
            patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg,
            patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_art_cls,
        ):
            mock_cfg.return_value = MINIMAL_CFG

            mock_model = MagicMock()
            mock_art_cls.return_value.fetch_model_artifact.return_value = (
                mock_model,
                "test_timestamp",
            )

            manager._load_model_artifact("calibration_model_X")

            mock_art_cls.return_value.fetch_model_artifact.assert_called_once_with(
                model_artifact_name="calibration_model_X"
            )


class TestPartialProjectionSliceOverflow:
    """P-03 (C-57): Partial projection must raise NotImplementedError."""

    def test_partial_projection_raises_not_implemented(self):
        """Partial projection must raise NotImplementedError with clear message."""
        handler = MagicMock()
        handler.shape = (10, 4, 4, 3)

        scaler = MagicMock()
        inference = MagicMock()
        inference.generate_posterior_samples.return_value = (
            np.zeros((5, 4, 4, 1, 2)),
            np.zeros((5, 4, 4, 1, 2)),
        )

        orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
        orchestrator.config = MINIMAL_CFG
        orchestrator.device = MagicMock()
        orchestrator.viz = MagicMock()
        orchestrator.viz.active = False

        with pytest.raises(NotImplementedError, match="Partial projection is not supported"):
            orchestrator._run_inference_pipeline(
                handler,
                scaler,
                inference,
                origin=7,
                origin_idx=0,
                is_backtest=True,
                n_origins=1,
                target_names=["lr_sb_best"],
            )


class TestForecastSnifferNeverCalled:
    """P-04 (C-58): Forecast pipeline must pass is_forecast=True to sniffer."""

    def test_forecast_mode_triggers_forecast_sniffer(self, tmp_path):
        """Forecast pipeline must validate temporal continuity."""
        import pandas as pd

        manager = _make_manager(tmp_path)

        df = pd.DataFrame(
            {
                "month_id": sorted(list(range(100, 112)) * 16),
                "priogrid_gid": list(range(1, 17)) * 12,
                "row": [0] * 192,
                "col": [0] * 192,
                "lr_sb_best": [1.0] * 192,
                "lr_ns_best": [1.0] * 192,
                "lr_os_best": [1.0] * 192,
            }
        )

        with (
            patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg,
            patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch_cls,
            patch("views_hydranet.manager.hydranet_manager.DataSniffer") as mock_sniffer_cls,
        ):
            mock_cfg.return_value = MINIMAL_CFG
            mock_fetch_cls.return_value.fetch_df.return_value = df
            mock_fetch_cls.standardize_raw_df.side_effect = lambda x, y: x

            mock_sniffer = mock_sniffer_cls.return_value
            mock_viz = MagicMock()

            manager._run_data_pipeline(mock_viz, forecast=True)

            mock_sniffer.sniff_forecast_alignment.assert_called_once()
            call_kwargs = mock_sniffer.sniff_forecast_alignment.call_args
            assert call_kwargs[1].get("is_forecast") is True or (
                len(call_kwargs[0]) >= 3 and call_kwargs[0][2] is True
            ), "sniff_forecast_alignment must receive is_forecast=True in forecast mode"
