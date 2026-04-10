from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch

from views_hydranet.manager.hydranet_manager import HydranetManager
from views_hydranet.train.train_model import train_model_artifact

# Minimal valid 3+3 config
SWEEP_CFG = {
    "run_type": "calibration",
    "sweep": True,
    "index_names": ["month_id", "priogrid_gid"],
    "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
    "classification_targets": ["by_sb", "by_ns", "by_os"],
    "features": ["f1", "f2", "f3"],
    "input_channels": 3,
    "output_channels": 1,
    "height": 4, "width": 4,
    "time_col": "month_id", "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "row_offset": 0, "col_offset": 0,
    "model": "Dummy", "window_dim": 1, "total_hidden_channels": 8,
    "dropout_rate": 0.0, "weight_init": "norm", "h_init": "zero",
    "learning_rate": 0.01, "weight_decay": 0.0, "windows_per_lesson": 1,
    "scheduler": "none", "warmup_steps": 1, "clip_grad_norm": True,
    "loss_reg": "mse", "loss_class": "bce",
    "loss_reg_a": 1, "loss_reg_c": 1, "loss_class_gamma": 1, "loss_class_alpha": 1,
    "total_lessons": 1, "n_posterior_samples": 1,
    "np_seed": 1, "torch_seed": 1,
    "min_events": 0, "slope_ratio": 0.1, "roof_ratio": 0.1, "max_ratio": 0.9, "min_ratio": 0.1,
    "freeze_h": "none", "evaluation_mode": "point", "aggregate_method": "arithmetic_mean",
    "steps": [1], "time_steps": 1,
    "identity_cols": ["month_id", "priogrid_gid"],
    "transformations": {"identity": ["f1", "f2", "f3", "lr_sb", "lr_ns", "lr_os"]},
}

def test_manager_sweep_skip_save_logic(tmp_path):
    """
    BEIGE GATE: Verify that in sweep mode, the artifact is NOT saved to disk.
    This ensures we don't nuke local storage during large sweeps.
    """
    mpm = MagicMock()
    mpm.artifacts = tmp_path

    # Mock dependencies to reach the save logic
    with (
        patch(
            "views_hydranet.train.train_model.make",
            return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock()),
        ),
        patch(
            "views_hydranet.train.train_model.training_loop",
            return_value={"final_loss": 0.1},
        ),
        patch("torch.save") as mock_save
    ):
        # 1. Standard Run (Save = True)
        cfg_std = SWEEP_CFG.copy()
        cfg_std["sweep"] = False
        _, _ = train_model_artifact(
            mpm, cfg_std, torch.device("cpu"),
            MagicMock(), save_artifact=True,
        )
        assert mock_save.called

        mock_save.reset_mock()

        # 2. Sweep Run (Save = False)
        cfg_sweep = SWEEP_CFG.copy()
        cfg_sweep["sweep"] = True
        _, _ = train_model_artifact(
            mpm, cfg_sweep, torch.device("cpu"),
            MagicMock(), save_artifact=False,
        )
        assert not mock_save.called

def test_manager_architecture_mismatch_red_gate():
    """
    RED GATE: Verify that the Preflight check kills the process if heads are misaligned.
    This protects against silent loss corruption.
    """
    bad_cfg = SWEEP_CFG.copy()
    bad_cfg["regression_targets"] = ["lr_sb"] # Only 1, architecture expects 3

    with patch(
        "views_pipeline_core.managers.model.model"
        ".ForecastingModelManager.__init__",
        return_value=None,
    ):
        manager = HydranetManager(model_path=MagicMock())
        with patch.object(HydranetManager, "configs", new_callable=PropertyMock) as mock_cfg:
            mock_cfg.return_value = bad_cfg

            with pytest.raises(ValueError, match="ARCHITECTURE MISMATCH"):
                manager._run_preflight_check()
