"""
Tests for InferenceOrchestrator.generate_prediction_frames().

generate_prediction_frames() is the sole output method:
- Inference pipeline sequence (Predict → Wrap → Invert → Collapse → to_pf)
- Returns list[dict[str, PredictionFrame]]
- No pandas object is materialised in the output path
- Per-origin memory release (del of large numpy arrays after each origin)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("views_pipeline_core")

from views_pipeline_core.data.prediction_frame import PredictionFrame

from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator
from views_hydranet.utils.volume_handler import VolumeHandler

# ─── Config & fixtures ───────────────────────────────────────────────────────

TARGETS = [
    "lr_sb_best",
    "lr_ns_best",
    "lr_os_best",
    "by_sb_best",
    "by_ns_best",
    "by_os_best",
]

ORCH_CFG = {
    "run_type": "calibration",
    "steps": [1],
    "time_steps": 1,
    "input_channels": 3,
    "output_channels": 1,
    "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    "identity_cols": ["row", "col"],
    "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "transformations": {"identity": ["lr_sb_best", "lr_ns_best", "lr_os_best"]},
    "derivations": {
        "binary": [
            {"from": "lr_sb_best", "to": "by_sb_best", "threshold": 0},
            {"from": "lr_ns_best", "to": "by_ns_best", "threshold": 0},
            {"from": "lr_os_best", "to": "by_os_best", "threshold": 0},
        ]
    },
    "height": 2,
    "width": 2,
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "index_names": ["month_id", "priogrid_gid"],
    "spatial_cols": ["row", "col"],
    "row_offset": 0,
    "col_offset": 0,
    "model": "HydraBNUNet06_LSTM4",
    "window_dim": 1,
    "total_hidden_channels": 8,
    "dropout_rate": 0.0,
    "weight_init": "norm",
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "windows_per_lesson": 1,
    "scheduler": "none",
    "warmup_steps": 1,
    "clip_grad_norm": True,
    "loss_reg": "mse",
    "loss_class": "bce",
    "loss_reg_a": 1,
    "loss_reg_c": 1,
    "loss_class_gamma": 1,
    "loss_class_alpha": 1,
    "total_lessons": 1,
    "n_posterior_samples": 4,
    "np_seed": 42,
    "torch_seed": 42,
    "min_events": 0,
    "slope_ratio": 0.1,
    "roof_ratio": 0.1,
    "max_ratio": 0.9,
    "min_ratio": 0.1,
    "freeze_h": "none",
    "evaluation_mode": "stochastic",
    "aggregate_method": "arithmetic_mean",
}

N_CELLS = 4
S = 4


def _make_df(n_months: int = 21) -> pd.DataFrame:
    rows = []
    for t in range(100, 100 + n_months):
        for r in range(2):
            for c in range(2):
                rows.append(
                    {
                        "month_id": t,
                        "priogrid_gid": r * 2 + c + 1,
                        "row": float(r),
                        "col": float(c),
                        "lr_sb_best": float(r + c + t * 0.01),
                        "lr_ns_best": 0.5,
                        "lr_os_best": 0.0,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def orch_env():
    """Shared handler + scaler used across orchestrator tests."""
    df = _make_df(n_months=21)
    handler = VolumeHandler.from_df(df, ORCH_CFG)
    scaler = FeatureScaler(ORCH_CFG)
    scaler._is_fitted = True
    model = MagicMock(spec=torch.nn.Module)
    return handler, scaler, model


# ─── Tests ───────────────────────────────────────────────────────────────────


class TestGreen:
    """Green: InferenceOrchestrator.generate_prediction_frames() contract."""

    def test_returns_list_of_dicts(self, orch_env):
        """Return type is list[dict[str, PredictionFrame]]."""
        handler, scaler, model = orch_env
        np.random.seed(0)
        posterior = np.random.rand(1, 2, 2, 6, S)

        cfg = {**ORCH_CFG, "evaluation_mode": "stochastic"}
        orchestrator = InferenceOrchestrator(cfg, model, torch.device("cpu"))

        with patch(
            "views_hydranet.utils.inference_orchestrator.HydraNetInference"
        ) as mock_inf_cls:
            mock_inf_cls.return_value.generate_posterior_samples.return_value = (posterior, None)
            origins = [handler.shape[0] - 2]
            result = orchestrator.generate_prediction_frames(
                handler, scaler, origins=origins, all_targets=TARGETS
            )

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        for target in TARGETS:
            assert target in result[0]
            assert isinstance(result[0][target], PredictionFrame)

    def test_stochastic_shape(self, orch_env):
        """Stochastic mode: every PF has y_pred.shape == (N_CELLS, S)."""
        handler, scaler, model = orch_env
        np.random.seed(1)
        posterior = np.random.rand(1, 2, 2, 6, S)

        cfg = {**ORCH_CFG, "evaluation_mode": "stochastic"}
        orchestrator = InferenceOrchestrator(cfg, model, torch.device("cpu"))

        with patch(
            "views_hydranet.utils.inference_orchestrator.HydraNetInference"
        ) as mock_inf_cls:
            mock_inf_cls.return_value.generate_posterior_samples.return_value = (posterior, None)
            result = orchestrator.generate_prediction_frames(
                handler, scaler, origins=[handler.shape[0] - 2], all_targets=TARGETS
            )

        for target in TARGETS:
            pf = result[0][target]
            assert pf.y_pred.shape == (N_CELLS, S), (
                f"{target}: expected ({N_CELLS}, {S}), got {pf.y_pred.shape}"
            )

    def test_point_shape(self, orch_env):
        """Point mode: every PF has y_pred.shape == (N_CELLS, 1)."""
        handler, scaler, model = orch_env
        np.random.seed(2)
        posterior = np.random.rand(1, 2, 2, 6, S)

        cfg = {**ORCH_CFG, "evaluation_mode": "point", "aggregate_method": "arithmetic_mean"}
        orchestrator = InferenceOrchestrator(cfg, model, torch.device("cpu"))

        with patch(
            "views_hydranet.utils.inference_orchestrator.HydraNetInference"
        ) as mock_inf_cls:
            mock_inf_cls.return_value.generate_posterior_samples.return_value = (posterior, None)
            result = orchestrator.generate_prediction_frames(
                handler, scaler, origins=[handler.shape[0] - 2], all_targets=TARGETS
            )

        for target in TARGETS:
            pf = result[0][target]
            assert pf.y_pred.shape == (N_CELLS, 1), (
                f"{target}: expected ({N_CELLS}, 1), got {pf.y_pred.shape}"
            )

    def test_multiple_origins(self, orch_env):
        """Rolling-origin evaluation: list length equals number of origins."""
        handler, scaler, model = orch_env
        np.random.seed(3)
        posterior = np.random.rand(1, 2, 2, 6, S)

        cfg = {**ORCH_CFG, "evaluation_mode": "stochastic"}
        orchestrator = InferenceOrchestrator(cfg, model, torch.device("cpu"))

        n_origins = 3
        origins = list(range(handler.shape[0] - n_origins - 1, handler.shape[0] - 1))

        with patch(
            "views_hydranet.utils.inference_orchestrator.HydraNetInference"
        ) as mock_inf_cls:
            mock_inf_cls.return_value.generate_posterior_samples.return_value = (posterior, None)
            result = orchestrator.generate_prediction_frames(
                handler, scaler, origins=origins, all_targets=TARGETS
            )

        assert len(result) == n_origins
