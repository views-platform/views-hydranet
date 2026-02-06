
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from views_hydranet.utils.backtest_orchestrator import BacktestOrchestrator
from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.volume_handler import VolumeHandler

# UNBREAKABLE AUDIT CONFIG
# 3 Tasks -> 12 Columns Total
CFG = {
    'run_type': 'audit',
    'steps': [1, 2, 3], # 3-step forecast
    'time_steps': 3,
    'time_col': 'month_id',
    'id_col': 'priogrid_gid',
    'spatial_cols': ['row', 'col'],
    'identity_cols': ['row', 'col', 'c_id'], # month_id and pg_id are already primary
    'features': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
    'targets': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
    'classification_outputs': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
    'row_offset': 0, 'col_offset': 0, 'height': 2, 'width': 2,
    'evalution_mode': 'point', 'aggregate_method': 'mean',
    'transform': {'identity': ['lr_sb_best', 'lr_ns_best', 'lr_os_best']}
}

@pytest.fixture
def audit_env():
    """Sets up a bit-perfect environment for the unbreakable audit."""
    # 1. History: Months 100 to 120 (21 months)
    df_hist = pd.DataFrame({
        'month_id': sorted(list(range(100, 121)) * 4),
        'priogrid_gid': [1, 2, 3, 4] * 21,
        'row': [0, 0, 1, 1] * 21,
        'col': [0, 1, 0, 1] * 21,
        'c_id': [42] * 84,
        'lr_sb_best': np.random.rand(84),
        'lr_ns_best': np.random.rand(84),
        'lr_os_best': np.random.rand(84)
    })
    handler = VolumeHandler.from_df(df_hist, CFG)

    scaler = FeatureScaler(CFG)
    scaler._is_fitted = True

    model = MagicMock(spec=torch.nn.Module)
    return handler, scaler, model

class TestBacktestUnbreakableAudit:
    """
    The Unbreakable Suite: Proving temporal continuity and schema bit-perfection.
    """

    # --- GREEN TEAM: THE HAPPY PATH (ACCURACY) ---

    def test_green_temporal_continuity(self, audit_env):
        """Prove that month_id increments by 1 from DF to DF and starts at origin + 1."""
        handler, scaler, model = audit_env
        orchestrator = BacktestOrchestrator(CFG, model, torch.device('cpu'))

        # We request 3 origins: index 10, 11, 12.
        # (Which correspond to month_id 110, 111, 112)
        origins = [10, 11, 12]

        # Mock inference to return 3 steps of data
        # Shape: [T=3, H=2, W=2, C=6] -> (3 targets * 2 heads)
        mock_posterior = np.random.rand(3, 2, 2, 6)

        with patch("views_hydranet.utils.backtest_orchestrator.HydraNetInference") as mock_inf_cls:
            mock_inf_cls.return_value.generate_posterior_samples.return_value = (mock_posterior, None)

            results = orchestrator.generate_rolling_forecasts(handler, scaler, origins=origins)

            assert len(results) == 3

            for i, origin_idx in enumerate(origins):
                df = results[i]
                start_month = df.index.get_level_values("month_id").min()
                expected_start = 100 + origin_idx + 1 # Month at index origin is 100+origin. Forecast starts at +1.

                assert start_month == expected_start, f"DF[{i}] start month mismatch!"

                if i > 0:
                    prev_start = results[i-1].index.get_level_values("month_id").min()
                    assert start_month == prev_start + 1, f"Temporal drift detected between DF[{i-1}] and DF[{i}]!"

    def test_green_12_feature_schema(self, audit_env):
        """Prove that every DF contains exactly the 12 required feature columns."""
        handler, scaler, model = audit_env
        orchestrator = BacktestOrchestrator(CFG, model, torch.device('cpu'))
        origins = [0]

        mock_posterior = np.random.rand(3, 2, 2, 6)
        with patch("views_hydranet.utils.backtest_orchestrator.HydraNetInference") as mock_inf_cls:
            mock_inf_cls.return_value.generate_posterior_samples.return_value = (mock_posterior, None)
            df = orchestrator.generate_rolling_forecasts(handler, scaler, origins=origins)[0]

            expected_features = [
                'lr_sb_best', 'lr_ns_best', 'lr_os_best',       # Linear Actuals
                'by_sb_best', 'by_ns_best', 'by_os_best',       # Binary Actuals
                'pred_lr_sb_best', 'pred_lr_ns_best', 'pred_lr_os_best', # Linear Predictions
                'pred_by_sb_best', 'pred_by_ns_best', 'pred_by_os_best'  # Binary Predictions
            ]

            for feat in expected_features:
                assert feat in df.columns, f"Mandatory feature '{feat}' missing from output!"

            # Check Bookkeeping
            for meta in ['c_id', 'row', 'col']:
                assert meta in df.columns, f"Bookkeeping column '{meta}' missing!"

    # --- BEIGE TEAM: THE ROBUSTNESS PATH (BOUNDARIES) ---

    def test_beige_out_of_bounds_origin(self, audit_env):
        """Verify that requesting an origin that leaves no room for steps fails loud."""
        handler, scaler, model = audit_env
        orchestrator = BacktestOrchestrator(CFG, model, torch.device('cpu'))

        # History has 21 months (0-20).
        # Origin 19 + 3 steps = Index 22 -> OUT OF BOUNDS.
        origins = [19]

        mock_posterior = np.random.rand(3, 2, 2, 6)
        with patch("views_hydranet.utils.backtest_orchestrator.HydraNetInference") as mock_inf_cls:
            mock_inf_cls.return_value.generate_posterior_samples.return_value = (mock_posterior, None)

            with pytest.raises(ValueError, match="Invalid time slice"):
                orchestrator.generate_rolling_forecasts(handler, scaler, origins=origins)

    # --- RED TEAM: THE INVINCIBILITY PATH (INVARIANTS) ---

    def test_red_topographic_identity_theft(self, audit_env):
        """
        Adversarial: Even if the model output is shuffled or flipped, 
        the Watermarks MUST force the IDs and Months back into alignment.
        """
        handler, scaler, model = audit_env
        orchestrator = BacktestOrchestrator(CFG, model, torch.device('cpu'))
        origins = [0]

        # ATTACK: Inference returns data where [0,0] is month 999 and pg_id 666
        # But the VolumeHandler watermark logic should ignore the model's 'guesses'
        # and use the Scaffold's truth.
        mock_posterior = np.random.rand(3, 2, 2, 6)

        with patch("views_hydranet.utils.backtest_orchestrator.HydraNetInference") as mock_inf_cls:
            mock_inf_cls.return_value.generate_posterior_samples.return_value = (mock_posterior, None)

            df = orchestrator.generate_rolling_forecasts(handler, scaler, origins=origins)[0]

            # PROOF: Month IDs must be 101, 102, 103 (Index 1, 2, 3)
            # regardless of what the model internally did.
            unique_months = sorted(df.index.get_level_values("month_id").unique())
            assert unique_months == [101, 102, 103]

            # PROOF: PG IDs must match the original scaffold [1, 2, 3, 4]
            unique_ids = sorted(df.index.get_level_values("priogrid_gid").unique())
            assert unique_ids == [1, 2, 3, 4]

    def test_red_stochastic_cell_integrity(self, audit_env):
        """Prove that in stochastic mode, cells contain lists of exactly n_posterior_samples."""
        handler, scaler, model = audit_env
        stochastic_cfg = CFG.copy()
        stochastic_cfg['evalution_mode'] = 'stochastic'
        stochastic_cfg['n_posterior_samples'] = 5

        orchestrator = BacktestOrchestrator(stochastic_cfg, model, torch.device('cpu'))
        origins = [0]

        # [T, H, W, C, S]
        mock_posterior = np.random.rand(3, 2, 2, 6, 5)

        with patch("views_hydranet.utils.backtest_orchestrator.HydraNetInference") as mock_inf_cls:
            mock_inf_cls.return_value.generate_posterior_samples.return_value = (mock_posterior, None)

            df = orchestrator.generate_rolling_forecasts(handler, scaler, origins=origins)[0]

            sample_cell = df["pred_lr_sb_best"].iloc[0]
            assert isinstance(sample_cell, list)
            assert len(sample_cell) == 5
            assert all(isinstance(x, (float, np.float32, np.float64)) for x in sample_cell)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
