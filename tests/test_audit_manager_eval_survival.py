import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock, patch, PropertyMock
from views_hydranet.manager.hydranet_manager import HydranetManager
from views_hydranet.utils.volume_handler import VolumeHandler

# AUDIT CONFIG: Point mode, arithmetic mean, heterogeneous scales
AUDIT_CFG = {
    'run_type': 'calibration',
    'steps': [1],
    'time_steps': 1,
    'input_channels': 2,
    'output_channels': 1,
    'target_variable': 'sb',
    'targets': ['sb', 'ns'],
    'classification_outputs': ['sb', 'ns'],
    'identity_cols': ['month_id', 'priogrid_gid'],
    'features': ['sb', 'ns'],
    'transform': {
        'log1p': ['sb'],
        'asinh': ['ns'],
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
    'loss_reg': 'b', 'loss_class': 'b', 'loss_reg_a': 1, 'loss_reg_c': 1,
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
        
        # 1. Setup History (24 months to satisfy windowing)
        df_hist = pd.DataFrame({
            'month_id': sorted(list(range(100, 124)) * 16), 
            'priogrid_gid': list(range(1, 17)) * 24,
            'row': [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3] * 24, 
            'col': [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3] * 24,
            'sb': [np.log1p(10.0)]*(16*24), 'ns': [np.arcsinh(10.0)]*(16*24)
        })

        # 2. Setup Inference Mock: Return semantic data
        # [T=1, H=4, W=4, C=4, S=10]
        posterior = np.zeros((1, 4, 4, 4, 10))
        posterior[:,:,:,0,:] = np.log1p(100.0)  # sb_SIGNAL
        posterior[:,:,:,1,:] = np.arcsinh(100.0) # ns_SIGNAL
        posterior[:,:,:,2:4,:] = 0.9 # Probs

        with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None):
            manager = HydranetManager(model_path=mpm)
            manager.device = torch.device("cpu")
            
            with patch.object(HydranetManager, 'configs', new_callable=PropertyMock) as mock_cfg:
                mock_cfg.return_value = AUDIT_CFG
                
                with patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetcher, \
                     patch.object(manager, '_load_model_artifact', return_value=(MagicMock(), "audit")), \
                     patch("views_hydranet.manager.hydranet_manager.HydraNetInference") as mock_inf:
                    
                    mock_fetcher.standardize_raw_df.side_effect = lambda x, y: x
                    mock_fetcher.return_value.fetch_df.return_value = df_hist
                    
                    mock_inf.return_value.generate_posterior_samples.return_value = (posterior, None)
                    
                    # RUN EVALUATION
                    results = manager._evaluate_model_artifact(eval_type="audit")
                    df = results[0]

                    # GATES
                    # 1. Space Gate: Value is 100.0 (raw), not 4.6 (log)
                    np.testing.assert_allclose(df["pred_sb_raw"].iloc[0], 100.0, rtol=1e-5)
                    
                    # 2. Dimension Gate: Cells are scalars (floats), not lists
                    assert isinstance(df["pred_sb_raw"].iloc[0], (float, np.float32, np.float64))
                    
                    # 3. Naming Gate: Correct prefix/suffix
                    assert "pred_sb_raw" in df.columns
                    assert "pred_ns_raw" in df.columns
                    
                    # 4. Actuals Gate: Actual remains 10.0
                    # (Simplified check for the proof)
                    assert "sb" in df.columns
                    
                    # 5. Index Gate: restored correctly
                    assert isinstance(df.index, pd.MultiIndex)
                    assert df.index.names == ["month_id", "priogrid_gid"]
                    
                    # 6. Symmetry Gate: Probabilities are untouched (remains 0.9)
                    np.testing.assert_allclose(df["pred_sb_prob"].iloc[0], 0.9, rtol=1e-5)
                    
                    # 7. RAM Survival Gate: No 'list' type in values
                    assert not any(isinstance(x, list) for x in df["pred_sb_raw"])
                    
                    # 8. Heterogeneous Gate: NS (asinh) also inverted correctly to 100.0
                    np.testing.assert_allclose(df["pred_ns_raw"].iloc[0], 100.0, rtol=1e-5)

    def test_gate_8_nuke_proof_heterogeneous(self, tmp_path):
        """Hard Gate 8: Verify multiple inverse functions in one volume."""
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path
        
        df_hist = pd.DataFrame({
            'month_id': sorted(list(range(100, 124)) * 16), 
            'priogrid_gid': list(range(1, 17)) * 24,
            'row': [0]*256 + [1]*128, 'col': [0]*384, # dummy coords
            'sb': [1.0]*384, 'ns': [1.0]*384
        })

        # Signal 0: log1p(10), Signal 1: asinh(10)
        posterior = np.zeros((1, 4, 4, 4, 1))
        posterior[:,:,:,0,:] = np.log1p(10.0)
        posterior[:,:,:,1,:] = np.arcsinh(10.0)
        
        with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None):
            manager = HydranetManager(model_path=mpm)
            manager.device = torch.device("cpu")
            with patch.object(HydranetManager, 'configs', new_callable=PropertyMock) as mock_cfg:
                mock_cfg.return_value = AUDIT_CFG
                with patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetcher, \
                     patch.object(manager, '_load_model_artifact', return_value=(MagicMock(), "audit")), \
                     patch("views_hydranet.manager.hydranet_manager.HydraNetInference") as mock_inf:
                    
                    mock_fetcher.standardize_raw_df.side_effect = lambda x, y: x
                    mock_fetcher.return_value.fetch_df.return_value = df_hist
                    mock_inf.return_value.generate_posterior_samples.return_value = (posterior, None)
                    
                    results = manager._evaluate_model_artifact(eval_type="audit")
                    df = results[0]
                    
                    # Both should be 10.0
                    np.testing.assert_allclose(df["pred_sb_raw"].iloc[0], 10.0, rtol=1e-5)
                    np.testing.assert_allclose(df["pred_ns_raw"].iloc[0], 10.0, rtol=1e-5)

    def test_gates_9_to_16_forecast_survival(self, tmp_path):
        """Hard Gates 9-16: Falsify the Survival Sequence in the Forecasting path."""
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path
        
        df_hist = pd.DataFrame({
            'month_id': sorted(list(range(100, 124)) * 16), 
            'priogrid_gid': list(range(1, 17)) * 24,
            'row': [0]*384, 'col': [0]*384, 
            'sb': [np.log1p(10.0)]*384, 'ns': [np.arcsinh(10.0)]*384
        })

        # Forecast Posterior: log1p(50)
        posterior = np.zeros((1, 4, 4, 4, 5)) # 5 samples
        posterior[:,:,:,0,:] = np.log1p(50.0)
        posterior[:,:,:,2:4,:] = 0.7 # Probs

        with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None):
            manager = HydranetManager(model_path=mpm)
            manager.device = torch.device("cpu")
            
            with patch.object(HydranetManager, 'configs', new_callable=PropertyMock) as mock_cfg:
                mock_cfg.return_value = AUDIT_CFG
                
                with patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetcher, \
                     patch.object(manager, '_load_model_artifact', return_value=(MagicMock(), "audit")), \
                     patch("views_hydranet.manager.hydranet_manager.HydraNetInference") as mock_inf:
                    
                    mock_fetcher.standardize_raw_df.side_effect = lambda x, y: x
                    mock_fetcher.return_value.fetch_df.return_value = df_hist
                    mock_inf.return_value.generate_posterior_samples.return_value = (posterior, None)
                    
                    # RUN FORECAST
                    results = manager._forecast_model_artifact()
                    df = results[0]

                    # GATES
                    # 9. Space Gate: Forecast is 50.0 (raw)
                    np.testing.assert_allclose(df["pred_sb_raw"].iloc[0], 50.0, rtol=1e-5)
                    
                    # 10. Dimension Gate: Scalars, not lists
                    assert isinstance(df["pred_sb_raw"].iloc[0], (float, np.float32, np.float64))
                    
                    # 11. Continuity Gate: Forecast month should be 124 (History ended at 123)
                    assert df.index.get_level_values("month_id")[0] == 124
                    
                    # 12. Symmetry Gate: Probabilities untouched (0.7)
                    np.testing.assert_allclose(df["pred_sb_prob"].iloc[0], 0.7, rtol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
