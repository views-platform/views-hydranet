import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pydantic import ValidationError
from views_hydranet.utils.utils_config import HydraNetConfig, TRANSFORMS
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.manager.hydranet_manager import HydranetManager
from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4

# ROOT FIX AUDIT STANDARDS
BASE_CONFIG = {
    'run_type': 'calibration',
    'steps': [1, 2],
    'time_steps': 2,
    'input_channels': 3,
    'output_channels': 1,
    'target_variable': 'lr_sb_best',
    'targets': ['lr_sb_best'],
    'classification_outputs': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
    'identity_cols': ['month_id', 'priogrid_gid', 'row', 'col'],
    'features': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
    'log1p': ['lr_sb_best'],
    'asinh': ['lr_ns_best'],
    'identity': ['lr_os_best'],
    'height': 4, 'width': 4,
    'time_col': 'month_id', 'id_col': 'priogrid_gid',
    'spatial_cols': ['row', 'col'],
    'row_offset': 0, 'col_offset': 0,
    'model': 'HydraBNUNet06_LSTM4',
    'window_dim': 2,
    'total_hidden_channels': 16,
    'dropout_rate': 0.0,
    'weight_init': 'xavier_norm',
    'h_init': 'abs_rand_exp-100',
    'learning_rate': 0.001,
    'weight_decay': 0.0,
    'windows_per_lesson': 1,
    'scheduler': 'WarmupDecay',
    'warmup_steps': 1,
    'clip_grad_norm': True,
    'loss_reg': 'b', 'loss_class': 'b',
    'loss_reg_a': 1, 'loss_reg_c': 1,
    'loss_class_gamma': 1, 'loss_class_alpha': 1,
    'total_lessons': 1,
    'n_posterior_samples': 1,
    'np_seed': 4, 'torch_seed': 4,
    'min_events': 0, 'slope_ratio': 0.5, 'roof_ratio': 0.5,
    'max_ratio': 0.9, 'min_ratio': 0.1,
    'freeze_h': 'hl',
    'evalution_mode': 'point',
    'aggregate_method': 'geometric_mean'
}

class TestNukeProofAudit:
    """The 36-Gate Nuke-Proof Falsification Suite."""

    # --- ZONE 1: CONFIG HANDSHAKE ---
    
    @pytest.mark.parametrize("missing_key", ["h_init", "weight_init", "output_channels", "clip_grad_norm"])
    def test_gate_1_to_4_strictness(self, missing_key):
        bad_cfg = BASE_CONFIG.copy()
        del bad_cfg[missing_key]
        with pytest.raises(ValidationError):
            HydraNetConfig(**bad_cfg)

    def test_gate_5_scaling_law_missing_feature(self):
        bad_cfg = BASE_CONFIG.copy()
        bad_cfg['log1p'] = []
        bad_cfg['asinh'] = []
        bad_cfg['identity'] = []
        with pytest.raises(ValidationError, match="not assigned a transform"):
            HydraNetConfig(**bad_cfg)

    def test_gate_8_checksum_input_channels(self):
        bad_cfg = BASE_CONFIG.copy()
        bad_cfg['input_channels'] = 99
        with pytest.raises(ValidationError, match="Checksum Law Violation: input_channels"):
            HydraNetConfig(**bad_cfg)

    def test_gate_12_baggage_tolerance(self):
        baggage_cfg = BASE_CONFIG.copy()
        baggage_cfg['deployment_status'] = 'shadow'
        cfg = HydraNetConfig(**baggage_cfg)
        assert cfg.model_dump()['deployment_status'] == 'shadow'

    def test_gate_13_14_typo_normalization(self):
        typo_cfg = BASE_CONFIG.copy()
        typo_cfg['evalution_mode'] = 'stocastic'
        typo_cfg['aggregate_method'] = 'mean'
        cfg = HydraNetConfig(**typo_cfg)
        assert cfg.evalution_mode == 'stochastic'
        assert cfg.aggregate_method == 'geometric_mean'

    # --- ZONE 2: VOLUME PHYSICS ---

    def test_gate_15_16_identity_striping(self):
        df = pd.DataFrame({
            'month_id': [1, 1], 'priogrid_gid': [1, 2],
            'row': [0, 0], 'col': [0, 1],
            'lr_sb_best': [1.0, 2.0], 'lr_ns_best': [0.0, 0.0], 'lr_os_best': [0.0, 0.0]
        })
        handler = VolumeHandler.from_df(df, BASE_CONFIG, height=4, width=4)
        tensor = handler.to_pytorch(torch.device('cpu'), include_identities=False)
        assert tensor.shape == (1, 1, 3, 4, 4) 

    def test_gate_20_21_22_23_symmetry_recovery(self):
        posterior = torch.ones((1, 1, 6, 4, 4))
        df_hist = pd.DataFrame({
            'month_id': [1, 1], 'priogrid_gid': [1, 2],
            'row': [0, 0], 'col': [0, 1],
            'lr_sb_best': [10.0, 20.0], 'lr_ns_best': [0.0, 0.0], 'lr_os_best': [0.0, 0.0]
        })
        handler = VolumeHandler.from_df(df_hist, BASE_CONFIG, height=4, width=4)
        pred_handler = handler.wrap_predictions(posterior, base_names=BASE_CONFIG['classification_outputs'])
        df_res = pred_handler.to_evaluation_df(history=handler, start_idx=0)
        
        assert isinstance(df_res.index, pd.MultiIndex)
        assert df_res.index.names == ["month_id", "priogrid_gid"]
        assert "lr_sb_best" in df_res.columns
        assert "pred_lr_sb_best_raw" in df_res.columns

    # --- ZONE 3: MATH ---

    def test_gate_26_27_28_math_precision(self):
        scaler = FeatureScaler(BASE_CONFIG)
        df = pd.DataFrame({
            'lr_sb_best': [10.0, 100.0, 1000.0],
            'lr_ns_best': [10.0, 100.0, 1000.0],
            'lr_os_best': [10.0, 100.0, 1000.0]
        })
        semantic = scaler.fit_transform(df)
        recovered = scaler.inverse_transform(semantic)
        for col in df.columns:
            np.testing.assert_allclose(df[col], recovered[col], rtol=1e-6)

    def test_gate_29_30_scaler_gate_law(self):
        scaler = FeatureScaler(BASE_CONFIG)
        df = pd.DataFrame({'lr_sb_best': [1.0], 'lr_ns_best': [1.0], 'lr_os_best': [1.0]})
        with pytest.raises(RuntimeError, match="Must be FITTED"):
            scaler.inverse_transform(df)
        scaler.fit_transform(df)
        with pytest.raises(RuntimeError, match="already fitted"):
            scaler.fit_transform(df)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])