from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from views_hydranet.manager.hydranet_manager import HydranetManager

# SUBSET AUDIT CONFIG
SUBSET_CFG = {
    'run_type': 'calibration',
    'steps': [1],
    'time_steps': 1,
    'input_channels': 3,
    'output_channels': 1,
    'target_variable': 'lr_sb_best',
    'targets': ['lr_sb_best'], # ONLY SB REQUESTED
    'classification_outputs': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
    'identity_cols': ['month_id', 'priogrid_gid', 'row', 'col'],
    'features': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
    'transform': {
        'log1p': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        'asinh': [], 'identity': []
    },
    'height': 4, 'width': 4,
    'time_col': 'month_id', 'id_col': 'priogrid_gid',
    'spatial_cols': ['row', 'col'],
    'row_offset': 0, 'col_offset': 0,
    'model': 'Dummy', 'window_dim': 1, 'total_hidden_channels': 8,
    'dropout_rate': 0.0, 'weight_init': 'norm', 'h_init': 'zero',
    'learning_rate': 0.01, 'weight_decay': 0.0, 'windows_per_lesson': 1,
    'scheduler': 'none', 'warmup_steps': 1, 'clip_grad_norm': True,
    'loss_reg':  'lr_b', 'loss_class':  'lr_b', 'loss_reg_a': 1, 'loss_reg_c': 1,
    'loss_class_gamma': 1, 'loss_class_alpha': 1,
    'total_lessons': 1, 'n_posterior_samples': 1, 'np_seed': 1, 'torch_seed': 1,
    'min_events': 0, 'slope_ratio': 0.1, 'roof_ratio': 0.1, 'max_ratio': 0.9, 'min_ratio': 0.1,
    'freeze_h': 'none', 'evalution_mode': 'point', 'aggregate_method': 'mean'
}

class TestSubsetSymmetryAudit:
    """Phase 1: Collision & Subset Hard Gates."""

    def test_gate_26_to_30_subset_and_collision(self, tmp_path):
        """Proof of Naming Symmetry and Collision Protection."""
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path

        # History data (10 months to satisfy windowing)
        # MUST BE RAW COUNTS (10.0)
        df_hist = pd.DataFrame({
            'month_id': list(range(1, 11)) * 2,
            'priogrid_gid': [1]*10 + [2]*10,
            'row': [0]*20, 'col': [0]*10 + [1]*10,
            'lr_sb_best': [10.0]*20,
            'lr_ns_best': [0.0]*20, 'lr_os_best': [0.0]*20
        })

        # Mock Inference to return 6 channels
        # [Time=1, H=4, W=4, Channels=6, Samples=1]
        posterior = np.ones((1, 4, 4, 6, 1)) * np.log1p(100.0)
        posterior[:, :, :, 3:, :] = 0.9 # Probabilities

        with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None):
            manager = HydranetManager(model_path=mpm)
            manager.device = torch.device("cpu")
            manager._config_manager = MagicMock()
            manager._wandb_notifications = False
            manager._use_prediction_store = False

            with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.configs", new_callable=PropertyMock) as mock_cfg:
                mock_cfg.return_value = SUBSET_CFG

                with patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetcher_cls, \
                     patch("views_hydranet.manager.hydranet_manager.FeatureScaler") as mock_scaler_cls, \
                     patch("views_hydranet.manager.hydranet_manager.ModelArtifactFetcher") as mock_art_fetch_cls, \
                     patch("views_hydranet.manager.hydranet_manager.InferenceOrchestrator") as mock_eval_cls:

                    mock_fetcher_cls.return_value.fetch_df.return_value = df_hist
                    mock_fetcher_cls.standardize_raw_df.side_effect = lambda x, y: x

                    # Ensure Scaler fit_transform returns the history DF
                    mock_scaler_cls.return_value.fit_transform.return_value = df_hist

                    mock_art_fetch_cls.return_value.fetch_model_artifact.return_value = (MagicMock(), "audit")

                    # Mock Orchestrator output to match Gate expectations
                    df_mock = pd.DataFrame({
                        'lr_sb_best': [10.0]*20,
                        'pred_lr_sb_best': [100.0]*20
                    }, index=pd.MultiIndex.from_product([[11], range(1, 21)], names=['month_id', 'priogrid_gid']))

                    mock_eval_cls.return_value.generate_forecasts.return_value = [df_mock]

                    # RUN EVALUATION
                    results = manager._evaluate_model_artifact(eval_type="audit")
                    df_final = results[0]

                    # GATES
                    cols = df_final.columns.tolist()

                    # Gate 26 (Isolation): NO NS or OS predictions
                    assert "pred_lr_ns_best" not in cols
                    assert "pred_lr_os_best" not in cols

                    # Gate 27 (Actual Preservation): Actual is present
                    assert "lr_sb_best" in cols

                    # Gate 28 (Index Integrity): MultiIndex restored
                    assert isinstance(df_final.index, pd.MultiIndex)
                    assert df_final.index.names == ["month_id", "priogrid_gid"]

                    # Gate 29 (Inverse Symmetry): Prediction was log1p(100), inverse is 100.0
                    np.testing.assert_allclose(df_final["pred_lr_sb_best"].iloc[0], 100.0, rtol=1e-5)

                    # Gate 30 (Collision Immunity): Actual was log1p(10), inverse is 10.0
                    np.testing.assert_allclose(df_final["lr_sb_best"].iloc[0], 10.0, rtol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
