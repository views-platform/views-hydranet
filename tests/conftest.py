import pytest
import torch
import torch.nn as nn

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import ModelOutput

# --- Minimum test count gate ---
# Catches silent test collection failures (e.g., broken imports from
# views_pipeline_core) before they erode coverage unnoticed.
# Only triggers when running the full suite (no specific test paths given).
# Assumes views_pipeline_core is installed (full monorepo environment).
# Without it, ~13 test files fail to collect, reducing count to ~270.
_MINIMUM_EXPECTED_TESTS = 287


def pytest_collection_modifyitems(config, items):
    # Only enforce when running the full test directory (no specific files/paths).
    # Bare `pytest` (no args) is treated as full-suite invocation.
    args = config.invocation_params.args
    positional = [a for a in args if not a.startswith("-")]
    is_full_suite = not positional or all(a in ("tests/", "tests") for a in positional)
    if is_full_suite and len(items) < _MINIMUM_EXPECTED_TESTS:
        raise pytest.UsageError(
            f"Test collection gate: only {len(items)} tests collected, "
            f"expected at least {_MINIMUM_EXPECTED_TESTS}. "
            f"Check for import errors in test files."
        )


class TinyModel(nn.Module):
    """
    Minimal model matching HydraNet's interface contract:
    forward(x, h) -> (reg_pred, cls_pred, h_next)

    x: [B, C_in, H, W]
    h: [B, hidden, H, W]
    Returns: reg [B, n_reg, H, W], cls [B, n_cls, H, W], h [B, hidden, H, W]
    """

    def __init__(self, input_channels, n_reg, n_cls, hidden=4):
        super().__init__()
        self.base = hidden
        self.reg_head = nn.Conv2d(input_channels + hidden, n_reg, 1)
        self.cls_head = nn.Conv2d(input_channels + hidden, n_cls, 1)
        self.h_update = nn.Conv2d(input_channels + hidden, hidden, 1)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        return ModelOutput(
            reg=self.reg_head(combined),
            cls=self.cls_head(combined),
            h_next=self.h_update(combined),
        )

    def init_hTtime(self, hidden_channels, H, W):
        return torch.zeros(1, hidden_channels, H, W)


def tobit_config_3target(**overrides):
    """Shared 3-target Tobit config for tests needing HydraNetConfig validation.

    Used by: test_tobit_loss.py, test_per_target_sigma.py, test_learnable_sigma.py,
    test_scheduled_sampling.py, test_falsification_per_target_sigma.py.

    Override any field: tobit_config_3target(loss_reg_sigma={"lr_sb": 1.0, ...})
    """
    base = {
        "run_type": "calibration",
        "features": ["lr_sb", "lr_ns", "lr_os"],
        "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
        "classification_targets": ["by_sb", "by_ns", "by_os"],
        "height": 8,
        "width": 8,
        "index_names": ["month_id", "priogrid_gid"],
        "time_col": "month_id",
        "id_col": "priogrid_gid",
        "spatial_cols": ["row", "col"],
        "row_offset": 0,
        "col_offset": 0,
        "model": "HydraBNUNet06_LSTM4",
        "window_dim": 4,
        "total_hidden_channels": 16,
        "dropout_rate": 0.1,
        "weight_init": "xavier_norm",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "windows_per_lesson": 2,
        "scheduler": "plateau",
        "warmup_steps": 5,
        "clip_grad_norm": True,
        "loss_reg": "tobit",
        "loss_reg_sigma": 1.0,
        "loss_class": "bce",
        "total_lessons": 10,
        "n_posterior_samples": 5,
        "np_seed": 42,
        "torch_seed": 42,
        "min_events": 0,
        "slope_ratio": 1.0,
        "roof_ratio": 1.0,
        "max_ratio": 0.9,
        "min_ratio": 0.1,
        "freeze_h": "none",
        "sampling_strategy": "threshold",
        "evaluation_mode": "point",
        "aggregate_method": "arithmetic_mean",
        "prediction_format": "prediction_frame",
        "time_steps": 3,
        "steps": [1, 2, 3],
        "input_channels": 3,
        "output_channels": 1,
        "identity_cols": ["priogrid_gid", "month_id"],
        "transformations": {"log1p": ["lr_sb", "lr_ns", "lr_os"]},
        "derivations": {
            "binary": [
                {"from": "lr_sb", "to": "by_sb", "threshold": 0},
                {"from": "lr_ns", "to": "by_ns", "threshold": 0},
                {"from": "lr_os", "to": "by_os", "threshold": 0},
            ],
        },
    }
    base.update(overrides)
    return base


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
        "loss_reg": "mse",
        "loss_class": "bce",
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
        "height": 180,
        "width": 360,
        "index_names": ["month_id", "priogrid_gid"],
        "row_offset": 0,
        "col_offset": 0,
        "max_ratio": 0.9,
        "min_ratio": 0.1,
        "time_steps": 36,
        "sampling_strategy": "threshold",
        "evaluation_mode": "point",
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
