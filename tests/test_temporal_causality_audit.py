"""
Temporal causality audit tests (ADR-005 taxonomy).

Green: output count matches config.
Beige: variable duration handling.
Red: causal shielding (poison attack), partition alignment.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import ModelOutput
from views_hydranet.utils.hydranet_inference import HydraNetInference


class CausalMockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = 32

    def init_hTtime(self, hidden_channels, H, W):
        return torch.zeros(1, hidden_channels, H, W)

    def forward(self, x, h):
        t1_pred = x + 0.1
        t1_cls = torch.zeros_like(t1_pred)
        h_new = h + 0.01
        return ModelOutput(reg=t1_pred, cls=t1_cls, h_next=h_new)


@pytest.fixture
def causal_setup():
    model = CausalMockModel()
    cfg = {
        "steps": list(range(1, 37)),
        "time_steps": 36,
        "features": ["feat"],
        "regression_targets": ["feat"],
        "classification_targets": ["class"],
        "freeze_h": "none",
        "sampling_strategy": "threshold",
        "n_posterior_samples": 1,
        "diagnostic_visualizations": False,
    }
    return model, cfg


class TestGreen:
    """Green: Temporal count integrity."""

    def test_temporal_count_integrity(self, causal_setup):
        """Verify that output matches config count exactly."""
        model, cfg = causal_setup
        inf = HydraNetInference(model, cfg, device="cpu")

        H, W = 4, 4
        full_tensor = torch.ones(1, 10, 1, H, W)
        feature_names = ["feat"]

        handler = MagicMock()
        handler.channel_map = feature_names
        handler.feature_cols = feature_names

        magnitudes, _ = inf.predict(full_tensor, 9, 0, feature_names)

        assert magnitudes.shape[0] == 36, f"Expected 36 months, got {magnitudes.shape[0]}"
        assert magnitudes.shape[1] == 1, "Expected 1 regression target"


class TestBeige:
    """Beige: Variable duration handling."""

    def test_variable_duration_beige_gate(self, causal_setup):
        """Verify that the system handles various time_steps without drift."""
        model, cfg = causal_setup
        inf = HydraNetInference(model, cfg, device="cpu")
        feature_names = ["feat"]
        full_tensor = torch.ones(1, 10, 1, 4, 4)

        for steps in [1, 12, 48]:
            inf.config["time_steps"] = steps
            inf.config["steps"] = list(range(1, steps + 1))

            mag, _ = inf.predict(full_tensor, 9, 0, feature_names)
            assert mag.shape[0] == steps


class TestRed:
    """Red: Causal isolation and partition alignment."""

    def test_causal_shielding_poison_attack(self, causal_setup):
        """Prove that changing ground truth in the 'future' has zero effect on predictions."""
        model, cfg = causal_setup
        inf = HydraNetInference(model, cfg, device="cpu")

        H, W = 4, 4
        feature_names = ["feat"]

        ten_month_tensor = torch.ones(1, 10, 1, H, W)
        mag_base, _ = inf.predict(ten_month_tensor, 9, 0, feature_names)

        clean_tensor_20 = torch.ones(1, 20, 1, H, W)
        poison_tensor_20 = clean_tensor_20.clone()
        poison_tensor_20[:, 10:, :, :, :] = 999.0

        mag_clean, _ = inf.predict(clean_tensor_20, 9, 0, feature_names)
        mag_poison, _ = inf.predict(poison_tensor_20, 9, 0, feature_names)

        assert np.allclose(mag_clean, mag_poison), "Leakage! Future data affected prediction."
        assert np.allclose(mag_clean, mag_base), "Drift! Tensor length affected prediction."

        clean_tensor_20 = torch.ones(1, 20, 1, H, W)
        poison_endpoint = clean_tensor_20.clone()
        poison_endpoint[:, 19, :, :, :] = 777.0

        res_clean, _ = inf.predict(clean_tensor_20, 19, 0, feature_names)
        res_poison, _ = inf.predict(poison_endpoint, 19, 0, feature_names)

        assert not np.allclose(res_clean[0], res_poison[0]), (
            "Model insensitive to history endpoint!"
        )

    def test_partition_alignment_purple_alien_scenario(self):
        """Prove the system naturally derives Origin 444 for 481-month partition with 36 steps."""
        from views_hydranet.utils.utils_orchestration import get_rolling_origin_indices

        total_months = 481
        time_steps = 36

        origins = get_rolling_origin_indices(total_months, time_steps, num_windows=1)
        assert origins == [444], f"Origin mechanics failure! Expected [444], got {origins}"

        model = CausalMockModel()
        cfg = {
            "steps": list(range(1, 37)),
            "time_steps": 36,
            "features": ["feat"],
            "regression_targets": ["feat"],
            "classification_targets": ["class"],
            "freeze_h": "none",
            "sampling_strategy": "threshold",
            "n_posterior_samples": 1,
            "diagnostic_visualizations": False,
        }

        inf = HydraNetInference(model, cfg, device="cpu")
        full_tensor = torch.zeros(1, total_months, 1, 4, 4)
        feature_names = ["feat"]

        magnitudes, _ = inf.predict(full_tensor, origins[0], 0, feature_names)
        assert magnitudes.shape[0] == 36, "Output length mismatch!"
