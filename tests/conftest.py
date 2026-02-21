from unittest.mock import MagicMock

import pytest


@pytest.fixture
def valid_config_dict():
    """
    Provides a complete, valid dictionary that satisfies the strict
    HydraNetConfig Pydantic model. These values are pulled from the
    current mission-ready defaults.
    """
    return {
        "model": "HydraBNUNet06_LSTM4",
        "run_type": "validation",
        "steps": list(range(1, 37)),
        "input_channels": 3,
        "output_channels": 1,
        "total_hidden_channels": 32,
        "dropout_rate": 0.125,
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "windows_per_lesson": 3,
        "scheduler": "WarmupDecay",
        "warmup_steps": 100,
        "total_lessons": 300,
        "n_posterior_samples": 10,
        "loss_reg": "lr_b",
        "loss_class": "lr_b",
        "loss_reg_a": 258,
        "loss_reg_c": 0.001,
        "loss_class_gamma": 1.5,
        "loss_class_alpha": 0.75,
        "weight_init": "xavier_norm",
        "clip_grad_norm": True,
        "min_events": 5,
        "slope_ratio": 0.75,
        "roof_ratio": 0.7,
        "freeze_h": "hl",
        "np_seed": 4,
        "torch_seed": 4,
        "window_dim": 32,
        "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
        "identity_cols": ["c_id", "row", "col"],
        "time_col": "month_id",
        "id_col": "priogrid_gid",
        "spatial_cols": ["row", "col"],
        "row_offset": 0,
        "col_offset": 0,
        "time_steps": 36,
        "evalution_mode": "point",
        "aggregate_method": "arithmetic_mean",
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
def mock_mpm(tmp_path):
    """Provides a mocked ModelPathManager with real temp paths."""
    mpm = MagicMock()
    mpm.logging = tmp_path / "logging"
    mpm.artifacts = tmp_path / "artifacts"
    mpm.data_processed = tmp_path / "data_processed"
    mpm.data_raw = tmp_path / "data_raw"
    mpm.models = tmp_path / "models"
    mpm.root = tmp_path / "root"
    mpm.data_generated = tmp_path / "generated"

    for d in [mpm.logging, mpm.artifacts, mpm.data_processed, mpm.data_raw, mpm.data_generated]:
        d.mkdir(parents=True, exist_ok=True)

    return mpm
