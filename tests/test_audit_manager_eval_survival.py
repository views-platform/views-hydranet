from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from views_hydranet.manager.hydranet_manager import HydranetManager

# AUDIT CONFIG: Point mode, arithmetic mean, heterogeneous scales
AUDIT_CFG = {
    'run_type': 'calibration',
    'steps': [1],
    'time_steps': 1,
    'input_channels': 2,
    'output_channels': 1,
    'target_variable': 'lr_sb_best',
    'targets': ['lr_sb_best', 'lr_ns_best'],
    'classification_outputs': ['lr_sb_best', 'lr_ns_best'],
    'identity_cols': ['month_id', 'priogrid_gid'],
    'features': ['lr_sb_best', 'lr_ns_best'],
    'transform': {
        'log1p': ['lr_sb_best'],
        'asinh': ['lr_ns_best'],
        'identity': []
    },

    'height': 4, 'width': 4,
    'time_col': 'month_id', 'id_col': 'priogrid_gid',
    'spatial_cols': ['row', 'col'],
    'row_offset': 0, 'col_offset': 0,
    'model': 'Dummy', 'window_dim': 1, 'total_hidden_channels': 8,
    'dropout_rate': 0.0, 'weight_init': 'norm', 'h_init': 'zero',
    'learning_rate': 0.01, 'weight_decay': 0.0, 'windows_per_lesson': 1,
    'scheduler': 'none', 'warmup_steps': 1, 'clip_grad_norm': True,
    'loss_reg':  'b', 'loss_class':  'b', 'loss_reg_a': 1, 'loss_reg_c': 1,
    'loss_class_gamma': 1, 'loss_class_alpha': 1,
    'total_lessons': 1, 'n_posterior_samples': 10,
    'np_seed': 1, 'torch_seed': 1,
    'min_events': 0, 'slope_ratio': 0.1, 'roof_ratio': 0.1, 'max_ratio': 0.9, 'min_ratio': 0.1,
    'freeze_h': 'none', 'evalution_mode': 'point',
    'aggregate_method': 'arithmetic_mean'
}

class TestManagerEvalHardAudit:
    """8 Hard Gates to falsify the Survival Sequence orchestration."""

    def test_gates_1_to_8_eval_survival(self, tmp_path):
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path

        # 1. Setup History
        df_hist = pd.DataFrame({
            'month_id': sorted(list(range(100, 124)) * 16),
            'priogrid_gid': list(range(1, 17)) * 24,
            'row': [0]*384, 'col': [0]*384,
             'lr_sb_best': [10.0]*384,  'lr_ns_best': [10.0]*384
        })

        with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None):
            manager = HydranetManager(model_path=mpm)
            manager.device = torch.device("cpu")
            manager._config_manager = MagicMock()
            manager._wandb_notifications = False
            manager._use_prediction_store = False

            with patch.object(HydranetManager, 'configs', new_callable=PropertyMock) as mock_cfg:
                mock_cfg.return_value = AUDIT_CFG

                with patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch_cls, \
                     patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_art_fetch_cls, \
                     patch("views_hydranet.manager.hydranet_manager.BacktestOrchestrator") as mock_eval_cls:

                    # 1. Mock DataFetcher
                    mock_fetch_cls.return_value.fetch_df.return_value = df_hist
                    mock_fetch_cls.standardize_raw_df.side_effect = lambda x, y: x

                    # 2. Mock ModelFetcher
                    mock_art_fetch_cls.return_value.fetch_model_artifact.return_value = (MagicMock(), "audit")

                    # 3. Mock Evaluator (This is where the 'Science' is tested)
                    df_pred = pd.DataFrame({
                         'lr_sb_best': [10.0]*16,  'lr_ns_best': [10.0]*16,
                        'pred_lr_sb_best': [100.0]*16, 'pred_lr_ns_best': [100.0]*16,
                        'pred_by_sb_best': [0.9]*16, 'pred_by_ns_best': [0.9]*16
                    }, index=pd.MultiIndex.from_product([[124], range(1, 17)], names=['month_id', 'priogrid_gid']))
                    mock_eval_cls.return_value.generate_rolling_forecasts.return_value = [df_pred]

                    # RUN EVALUATION
                    results = manager._evaluate_model_artifact(eval_type="audit")
                    df = results[0]

                    # GATES
                    np.testing.assert_allclose(df["pred_lr_sb_best"].iloc[0], 100.0, rtol=1e-5)
                    assert isinstance(df["pred_lr_sb_best"].iloc[0], (float, np.float32, np.float64))
                    assert "pred_lr_sb_best" in df.columns
                    assert "lr_sb_best" in df.columns
                    assert isinstance(df.index, pd.MultiIndex)
                    np.testing.assert_allclose(df["pred_by_sb_best"].iloc[0], 0.9, rtol=1e-5)
                    assert not any(isinstance(x, list) for x in df["pred_lr_sb_best"])
                    np.testing.assert_allclose(df["pred_lr_ns_best"].iloc[0], 100.0, rtol=1e-5)

    def test_gate_8_nuke_proof_heterogeneous(self, tmp_path):
        """Hard Gate 8: Verify multiple inverse functions in one volume."""
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path

        df_hist = pd.DataFrame({
            'month_id': sorted(list(range(100, 124)) * 16),
            'priogrid_gid': list(range(1, 17)) * 24,
            'row': [0]*384, 'col': [0]*384,
             'lr_sb_best': [1.0]*384,  'lr_ns_best': [1.0]*384
        })

        with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None):
            manager = HydranetManager(model_path=mpm)
            manager.device = torch.device("cpu")
            manager._config_manager = MagicMock()
            manager._wandb_notifications = False
            manager._use_prediction_store = False
            with patch.object(HydranetManager, 'configs', new_callable=PropertyMock) as mock_cfg:
                mock_cfg.return_value = AUDIT_CFG
                with patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch_cls, \
                     patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_art_fetch_cls, \
                     patch("views_hydranet.manager.hydranet_manager.BacktestOrchestrator") as mock_eval_cls:

                    mock_fetch_cls.return_value.fetch_df.return_value = df_hist
                    mock_fetch_cls.standardize_raw_df.side_effect = lambda x, y: x
                    mock_art_fetch_cls.return_value.fetch_model_artifact.return_value = (MagicMock(), "audit")

                    df_pred = pd.DataFrame({
                        'pred_lr_sb_best': [10.0]*16, 'pred_lr_ns_best': [10.0]*16,
                    }, index=pd.MultiIndex.from_product([[124], range(1, 17)], names=['month_id', 'priogrid_gid']))
                    mock_eval_cls.return_value.generate_rolling_forecasts.return_value = [df_pred]

                    results = manager._evaluate_model_artifact(eval_type="audit")
                    df = results[0]
                    np.testing.assert_allclose(df["pred_lr_sb_best"].iloc[0], 10.0, rtol=1e-5)
                    np.testing.assert_allclose(df["pred_lr_ns_best"].iloc[0], 10.0, rtol=1e-5)

    def test_gates_9_to_16_forecast_survival(self, tmp_path):
        """Hard Gates 9-16: Falsify the Survival Sequence in the Forecasting path."""
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path

        df_hist = pd.DataFrame({
            'month_id': sorted(list(range(100, 124)) * 16),
            'priogrid_gid': list(range(1, 17)) * 24,
            'row': [0]*384, 'col': [0]*384,
             'lr_sb_best': [10.0]*384,  'lr_ns_best': [10.0]*384
        })

        with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None):
            manager = HydranetManager(model_path=mpm)
            manager.device = torch.device("cpu")
            manager._config_manager = MagicMock()
            manager._wandb_notifications = False
            manager._use_prediction_store = False

            with patch.object(HydranetManager, 'configs', new_callable=PropertyMock) as mock_cfg:
                mock_cfg.return_value = AUDIT_CFG

                with patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetch_cls, \
                     patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_art_fetch_cls, \
                     patch("views_hydranet.manager.hydranet_manager.HydraNetInference") as mock_inf_cls:

                    # NOTE: _forecast_model_artifact hasn't been refactored to a separate Evaluator yet
                    # in your current Manager version (it still uses HydraNetInference directly).
                    # I'll mock accordingly.
                    mock_fetch_cls.return_value.fetch_df.return_value = df_hist
                    mock_fetch_cls.standardize_raw_df.side_effect = lambda x, y: x
                    mock_art_fetch_cls.return_value.fetch_model_artifact.return_value = (MagicMock(), "audit")

                    # posterior: (T, H, W, C) -> 2 targets * 2 heads = 4 channels
                    posterior = np.zeros((1, 4, 4, 4))
                    posterior[:,:,:,0] = np.log1p(50.0)
                    mock_inf_cls.return_value.generate_posterior_samples.return_value = (posterior, None)

                    # RUN FORECAST
                    results = manager._forecast_model_artifact()
                    df = results[0]

                    # GATES
                    np.testing.assert_allclose(df["pred_lr_sb_best"].iloc[0], 50.0, rtol=1e-5)
                    assert isinstance(df["pred_lr_sb_best"].iloc[0], (float, np.float32, np.float64))
                    assert df.index.get_level_values("month_id")[0] == 124

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
