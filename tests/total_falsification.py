from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
from views_hydranet.manager.hydranet_manager import HydranetManager
from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.utils_config import HydraNetConfig
from views_hydranet.utils.volume_handler import VolumeHandler

# BIT-PERFECT AUDIT STANDARDS (Root Fix Schema)
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
    'transform': {
        'log1p': ['lr_sb_best'],
        'asinh': ['lr_ns_best'],
        'identity': ['lr_os_best']
    },
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

class TestComprehensiveFalsificationAudit:
    """The Single Source of Truth for Bit-Perfect Architecture."""

    # --- ZONE 1: CONFIG HANDSHAKE ---

    @pytest.mark.parametrize("missing_key", ["h_init", "weight_init", "output_channels", "clip_grad_norm"])
    def test_gate_1_to_4_strictness(self, missing_key):
        bad_cfg = BASE_CONFIG.copy()
        del bad_cfg[missing_key]
        with pytest.raises(ValidationError):
            HydraNetConfig(**bad_cfg)

    def test_gate_5_scaling_law_missing_feature(self):
        bad_cfg = BASE_CONFIG.copy()
        bad_cfg['transform'] = {'log1p': ['lr_sb_best']}
        with pytest.raises(ValidationError, match="not assigned a transform"):
            HydraNetConfig(**bad_cfg)

    def test_gate_8_checksum_input_channels(self):
        bad_cfg = BASE_CONFIG.copy()
        bad_cfg['input_channels'] = 99
        with pytest.raises(ValidationError, match="Checksum Law Violation: input_channels"):
            HydraNetConfig(**bad_cfg)

    # --- ZONE 2: VOLUME PHYSICS & SYMMETRY ---

    def test_gate_15_identity_striping(self):
        df = pd.DataFrame({
            'month_id': [1, 1], 'priogrid_gid': [1, 2],
            'row': [0, 0], 'col': [0, 1],
            'lr_sb_best': [1.0, 2.0], 'lr_ns_best': [0.0, 0.0], 'lr_os_best': [0.0, 0.0]
        })
        handler = VolumeHandler.from_df(df, BASE_CONFIG, height=4, width=4)
        tensor = handler.to_pytorch(torch.device('cpu'), include_identities=False)
        assert tensor.shape == (1, 1, 3, 4, 4)

    def test_gate_20_21_22_symmetry_recovery(self, tmp_path):
        """Claim: Manager correctly names 6 heads and restores MultiIndex."""
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path
        model = HydraBNUNet06_LSTM4(3, 16, 1, 0.0)

        # History data
        df_hist = pd.DataFrame({
            'month_id': list(range(1, 11)) * 2,
            'priogrid_gid': [1]*10 + [2]*10,
            'row': [0]*20, 'col': [0]*10 + [1]*10,
            'lr_sb_best': [1.0]*20, 'lr_ns_best': [0.0]*20, 'lr_os_best': [0.0]*20
        })

        with patch("views_pipeline_core.managers.model.model.ForecastingModelManager.__init__", return_value=None):
            manager = HydranetManager(model_path=mpm)
            manager.device = torch.device("cpu")

            # HARDEN THE MANAGER CONFIG MOCK
            with patch.object(HydranetManager, 'configs', new_callable=PropertyMock) as mock_cfg:
                mock_cfg.return_value = BASE_CONFIG

                with patch("views_hydranet.manager.hydranet_manager.DataFetcher") as mock_fetcher, \
                     patch("views_hydranet.manager.hydranet_manager.FeatureScaler") as mock_scaler, \
                     patch.object(manager, '_load_model_artifact', return_value=(model, "audit")), \
                     patch("views_hydranet.manager.hydranet_manager.HydraNetInference") as mock_inf:

                    mock_fetcher.standardize_raw_df.side_effect = lambda x, y: x
                    mock_fetcher.return_value.fetch_df.return_value = df_hist
                    mock_scaler.return_value.fit_transform.return_value = df_hist
                    mock_scaler.return_value.inverse_transform.side_effect = lambda x: x

                    posterior = np.ones((1, 4, 4, 6, 1)) # 6 channels
                    mock_inf.return_value.generate_posterior_samples.return_value = (posterior, None)

                    results = manager._evaluate_model_artifact(eval_type="audit")
                    df_res = results[0]

                    assert isinstance(df_res.index, pd.MultiIndex)
                    assert df_res.index.names == ["month_id", "priogrid_gid"]
                    assert "pred_lr_sb_best_raw" in df_res.columns
                    assert "lr_sb_best" in df_res.columns # Actuals preserved

    # --- ZONE 3: MATH ---

    def test_gate_26_math_precision(self):
        scaler = FeatureScaler(BASE_CONFIG)
        df = pd.DataFrame({
            'lr_sb_best': [10.0, 100.0],
            'lr_ns_best': [10.0, 100.0],
            'lr_os_best': [10.0, 100.0]
        })
        semantic = scaler.fit_transform(df)
        recovered = scaler.inverse_transform(semantic)
        for col in df.columns:
            np.testing.assert_allclose(df[col], recovered[col], rtol=1e-6)

# --- ZONE 4: MEMORY & HARDWARE ---

    def test_gate_31_oom_sentinel(self):
        """Verify CUDA memory release after immediate backward."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for OOM Sentinel")

        device = torch.device("cuda")
        model = HydraBNUNet06_LSTM4(3, 16, 1, 0.0).to(device)
        x = torch.randn(1, 1, 3, 64, 64).to(device)
        h = model.init_hTtime(16, 64, 64).to(device).float()

        mem_start = torch.cuda.memory_allocated(device)
        reg, cl, _ = model(x[:, 0], h)
        loss = reg.sum()

        mem_with_graph = torch.cuda.memory_allocated(device)
        assert mem_with_graph > mem_start

        loss.backward()
        del loss, reg, cl, h, x

        mem_after = torch.cuda.memory_allocated(device)
        assert mem_after < mem_with_graph

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
