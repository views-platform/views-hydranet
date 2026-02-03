
import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import MagicMock, patch, PropertyMock
from views_hydranet.manager.hydranet_manager import HydranetManager
from views_hydranet.utils.utils_config import HydraNetConfig, TargetVariable
from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4

# Audit Configuration (The "Locked" standard)
AUDIT_CONFIG = {
    "run_type": "calibration",
    "steps": [1],
    "n_posterior_samples": 2,
    "total_lessons": 1,
    "windows_per_lesson": 3, # The "Mixed Salad"
    "input_channels": 3, 
    "transform": "log1p", 
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
    "h_init": "abs_rand_exp-100",
    "loss_reg": "b", 
    "loss_class": "b", 
    "loss_reg_a": 1, 
    "loss_reg_c": 1,
    "loss_class_gamma": 1, 
    "loss_class_alpha": 1, 
    "freeze_h": "hl",
    "evalution_mode": "stochastic", 
    "aggregate_method": "geometric_mean",
    "np_seed": 4, 
    "torch_seed": 4, 
    "min_events": 0, 
    "slope_ratio": 0.5, 
    "roof_ratio": 0.5,
    "max_ratio": 0.95, 
    "min_ratio": 0.05,
    "time_steps": 1,
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "row_offset": 0,
    "col_offset": 0,
    "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "identity_cols": ["month_id", "priogrid_gid", "row", "col"],
    "target_variable": "lr_sb_best",
    "classification_outputs": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "targets": ["lr_sb_best"],
    "transforms": {
        "log1p": ["lr_sb_best"],
        "asinh": ["lr_ns_best"],
        "identity": ["lr_os_best"]
    }
}
class TestPopperianAudit:
    """The Truth Engine: Falsifying Hallucination Claims."""

    def test_gate_1_symmetry_and_naming(self, tmp_path):
        """Claim: Manager correctly names 6 heads and protects Actuals."""
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path
        
        # Setup real model to verify head count
        model = HydraBNUNet06_LSTM4(3, 32, 1, 0.0)
        
        with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None):
            manager = HydranetManager(model_path=mpm)
            manager.device = torch.device("cpu")
            
            with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.configs", new_callable=PropertyMock) as mock_cfg:
                mock_configs = AUDIT_CONFIG.copy()
                mock_cfg.return_value = mock_configs
                
                # Mock components to isolate the Naming Engine
                with patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetcher:
                    df = pd.DataFrame({
                        "month_id": [100, 101], "priogrid_gid": [1, 1], 
                        "row": [0, 0], "col": [0, 0],
                        "lr_sb_best": [0.5, 0.6], "lr_ns_best": [0.1, 0.1], "lr_os_best": [0.0, 0.0]
                    })
                    mock_fetcher.return_value.fetch_df.return_value = df
                    mock_fetcher.standardize_raw_df.side_effect = lambda x, y: x
                    
                    with patch("views_hydranet.manager.hydranet_manager.FeatureScaler") as mock_scaler:
                        mock_scaler.return_value.fit_transform.side_effect = lambda x: x
                        mock_scaler.return_value.inverse_transform.side_effect = lambda x: x
                        mock_scaler.return_value.configured_columns = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
                        
                        with patch.object(manager, '_load_model_artifact', return_value=(model, "audit")):
                            results = manager._evaluate_model_artifact(eval_type="audit")
                            df_res = results[0]
                            
                            # FALSIFICATION GATES
                            cols = df_res.columns.tolist()
                            print(f"Audit Result Columns: {cols}")
                            
                            # 1. Did it produce 6 channels (3 reg, 3 class)? 
                            # (Targets filters NS/OS out, so we expect 2 pred + 1 actual)
                            assert "pred_lr_sb_best_raw" in cols, "Naming Engine failed prefix/suffix for Regression"
                            assert "pred_lr_sb_best_prob" in cols, "Naming Engine failed prefix/suffix for Classification"
                            
                            # 2. Did it preserve the Actual?
                            assert "lr_sb_best" in cols, "Actuals Protection failed: Actual was renamed or dropped"
                            
                            # 3. Did it subset correctly (NS and OS should be missing)?
                            assert "pred_lr_ns_best_raw" not in cols, "Subsetting Gate failed: included unrequested target"

    def test_gate_2_topology_index(self, tmp_path):
        """Claim: Final DF is correctly indexed as a spatiotemporal MultiIndex."""
        # Re-using logic from Gate 1 to get a DF
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path
        model = HydraBNUNet06_LSTM4(3, 32, 1, 0.0)
        with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None):
            manager = HydranetManager(model_path=mpm)
            manager.device = torch.device("cpu")
            with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.configs", new_callable=PropertyMock) as mock_cfg:
                mock_cfg.return_value = AUDIT_CONFIG
                
                # Real DF for Sniffer
                df = pd.DataFrame({
                    "month_id": [100, 101], "priogrid_gid": [1, 1], 
                    "row": [0, 0], "col": [0, 0],
                    "lr_sb_best": [0.5, 0.6], "lr_ns_best": [0.1, 0.1], "lr_os_best": [0.0, 0.0]
                })
                
                with patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetcher, \
                     patch("views_hydranet.manager.hydranet_manager.FeatureScaler") as mock_scaler, \
                     patch.object(manager, '_load_model_artifact', return_value=(model, "audit")):
                    
                    mock_fetcher.return_value.fetch_df.return_value = df
                    mock_fetcher.standardize_raw_df.side_effect = lambda x, y: x
                    mock_scaler.return_value.fit_transform.side_effect = lambda x: x
                    mock_scaler.return_value.inverse_transform.side_effect = lambda x: x
                    mock_scaler.return_value.configured_columns = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
                    
                    results = manager._evaluate_model_artifact(eval_type="audit")
                    df_res = results[0]
                    
                    # FALSIFICATION GATE
                    assert isinstance(df_res.index, pd.MultiIndex), "Topology failed: Output is not a MultiIndex"
                    assert df_res.index.names == ["month_id", "priogrid_gid"], f"Index names mismatch: {df_res.index.names}"

    def test_gate_3_memory_sentinel(self):
        """Claim: ADR 014 Hardening (Immediate Backward) clears graph memory."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory sentinel")
            
        device = torch.device("cuda")
        model = HydraBNUNet06_LSTM4(3, 32, 1, 0.0).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Initial memory with model loaded
        mem_start = torch.cuda.memory_allocated(device)
        
        # 1. Create a loss with a large graph
        x = torch.randn(1, 100, 3, 32, 32).to(device)
        h = model.init_hTtime(32, 32, 32).to(device).float()
        
        reg, cl, _ = model(x[:, 0], h)
        loss = reg.sum()
        
        mem_with_graph = torch.cuda.memory_allocated(device)
        assert mem_with_graph > mem_start, "Baseline: Graph must consume memory"
        
        # 2. IMMEDIATE BACKWARD (The Fix)
        loss.backward()
        
        mem_after_backward = torch.cuda.memory_allocated(device)
        # In PyTorch, memory isn't always released immediately to OS, but 
        # the graph nodes are freed. 
        # We assert that memory after backward is LESS than with graph
        assert mem_after_backward < mem_with_graph, "Memory Sentinel failed: Graph memory was not released after backward"

    def test_gate_4_boring_strictness(self):
        """Claim: Config is strict and rejects missing keys (ADR 008)."""
        bad_config = AUDIT_CONFIG.copy()
        del bad_config["identity_cols"]
        
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            HydraNetConfig(**bad_config)
            
        print("Boring Firewall: Correctly rejected incomplete config.")

if __name__ == "__main__":
    # If run directly, execute the audit
    pytest.main([__file__])
