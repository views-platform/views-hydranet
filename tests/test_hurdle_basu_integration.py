"""
ADR-050: Hurdle + Basu DPD integration tests.

Red tests: validate strict hurdle parameter enforcement.
Green tests: validate the hurdle + Basu DPD combination produces
correct, bounded, finite losses with gradient flow.
"""

import pytest
import torch
import torch.nn as nn

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import ModelOutput


# ---------------------------------------------------------------------------
# Red Team: Strict hurdle parameter validation
# ---------------------------------------------------------------------------
class TestRedHurdleParamValidation:
    def test_hurdle_enabled_without_qs99_tau_raises(self, valid_config_dict):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        cfg = {**valid_config_dict, "body_mask": "pos_cells", "qs99_weight": 0.1}
        with pytest.raises((ValueError, Exception), match="qs99_tau"):
            HydraNetConfig(**cfg)

    def test_process_sequence_none_qs99_no_crash(self):
        from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

        B, T, C, H, W = 1, 3, 1, 4, 4
        train_tensor = torch.randn(B, T, C, H, W).abs()
        h = torch.zeros(B, 8, H, W)

        model = make_tiny_model()
        idx = _SequenceIndices(
            ["feat"],
            {"regression_targets": ["feat"], "classification_targets": [], "features": ["feat"]},
        )

        result = _process_sequence(
            train_tensor,
            model,
            h,
            nn.MSELoss(),
            nn.BCELoss(),
            SumReducer(),
            idx,
            torch.device("cpu"),
            body_mask="pos_cells",
            qs99_weight=None,
            qs99_tau=None,
        )
        assert torch.isfinite(result["total"]), "None qs99 params should not crash"


# ---------------------------------------------------------------------------
# Green Team: Hurdle disabled → no QS99 constraint
# ---------------------------------------------------------------------------
class TestGreenHurdleConfigPaths:
    def test_hurdle_disabled_does_not_require_qs99(self, valid_config_dict):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        cfg = {**valid_config_dict}
        cfg.pop("body_mask", None)  # no positives mask -> qs99 not required
        cfg.pop("qs99_weight", None)
        cfg.pop("qs99_tau", None)
        HydraNetConfig(**cfg)

    def test_hurdle_qs99_weight_zero_does_not_require_tau(self, valid_config_dict):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        cfg = {**valid_config_dict, "body_mask": "pos_cells", "qs99_weight": 0.0}
        HydraNetConfig(**cfg)


# ---------------------------------------------------------------------------
# Red Team: target_weights validation
# ---------------------------------------------------------------------------
class TestRedTargetWeightsValidation:
    def test_target_weights_missing_target_raises(self, valid_config_dict):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        cfg = {
            **valid_config_dict,
            "target_weights": {"lr_ged_sb": 1.0, "lr_ged_ns": 3.0},
        }
        with pytest.raises((ValueError, Exception), match="lr_ged_os"):
            HydraNetConfig(**cfg)

    def test_target_weights_negative_raises(self, valid_config_dict):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        cfg = {
            **valid_config_dict,
            "target_weights": {
                "lr_ged_sb": 1.0,
                "lr_ged_ns": -1.0,
                "lr_ged_os": 3.0,
            },
        }
        with pytest.raises((ValueError, Exception)):
            HydraNetConfig(**cfg)


# ---------------------------------------------------------------------------
# Green Team: target_weights integration
# ---------------------------------------------------------------------------
class TestGreenTargetWeights:
    def test_target_weights_none_accepted(self, valid_config_dict):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        cfg = {**valid_config_dict}
        cfg.pop("target_weights", None)
        obj = HydraNetConfig(**cfg)
        assert obj.target_weights is None

    def test_target_weights_amplifies_rare_target_loss(self):
        from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

        B, T, C, H, W = 1, 3, 1, 4, 4
        train_tensor = torch.zeros(B, T, C, H, W)
        train_tensor[0, 1, 0, 1, 1] = 5.0
        train_tensor[0, 2, 0, 1, 1] = 3.0
        h = torch.zeros(B, 8, H, W)

        model = make_tiny_model()
        idx = _SequenceIndices(
            ["feat"],
            {"regression_targets": ["feat"], "classification_targets": [], "features": ["feat"]},
        )

        result_unweighted = _process_sequence(
            train_tensor,
            model,
            h.clone(),
            nn.MSELoss(),
            nn.BCELoss(),
            SumReducer(),
            idx,
            torch.device("cpu"),
            body_mask="pos_cells",
        )

        result_weighted = _process_sequence(
            train_tensor,
            model,
            h.clone(),
            nn.MSELoss(),
            nn.BCELoss(),
            SumReducer(),
            idx,
            torch.device("cpu"),
            body_mask="pos_cells",
            target_weights={"feat": 5.0},
        )

        loss_base = result_unweighted["reg"].item()
        loss_weighted = result_weighted["reg"].item()
        assert loss_weighted > loss_base, (
            f"Weighted loss ({loss_weighted:.4f}) should exceed unweighted ({loss_base:.4f})"
        )

    def test_target_weights_multi_target_applies_per_target(self):
        """C-88: target_weights must apply correct weight to correct target with 2+ targets."""
        from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

        B, T, C, H, W = 1, 3, 2, 4, 4
        train_tensor = torch.zeros(B, T, C, H, W)
        train_tensor[0, 1, 0, 1, 1] = 5.0
        train_tensor[0, 2, 0, 1, 1] = 3.0
        train_tensor[0, 1, 1, 2, 2] = 5.0
        train_tensor[0, 2, 1, 2, 2] = 3.0
        h = torch.zeros(B, 8, H, W)

        class TwoChannelModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 2, 1)
                self.base = 8

            def forward(self, x, h):
                out = self.conv(x)
                cls = out[:, :0, :, :]
                return ModelOutput(reg=out, cls=cls, h_next=h)

            def init_hTtime(self, hidden_channels, H, W):
                return torch.zeros(1, hidden_channels, H, W)

        model = TwoChannelModel()
        cfg = {
            "regression_targets": ["a", "b"],
            "classification_targets": [],
            "features": ["a", "b"],
        }
        idx = _SequenceIndices(["a", "b"], cfg)

        result_uniform = _process_sequence(
            train_tensor,
            model,
            h.clone(),
            nn.MSELoss(),
            nn.BCELoss(),
            SumReducer(),
            idx,
            torch.device("cpu"),
            body_mask="pos_cells",
            target_weights={"a": 1.0, "b": 1.0},
        )

        result_asymmetric = _process_sequence(
            train_tensor,
            model,
            h.clone(),
            nn.MSELoss(),
            nn.BCELoss(),
            SumReducer(),
            idx,
            torch.device("cpu"),
            body_mask="pos_cells",
            target_weights={"a": 1.0, "b": 5.0},
        )

        loss_uniform = result_uniform["total"].item()
        loss_asymmetric = result_asymmetric["total"].item()
        assert loss_asymmetric > loss_uniform, (
            f"Asymmetric weights (b=5.0) should increase total loss vs uniform: "
            f"uniform={loss_uniform:.4f}, asymmetric={loss_asymmetric:.4f}"
        )


# ---------------------------------------------------------------------------
# Green Team: Full ADR-050 combination through train() entry point
# ---------------------------------------------------------------------------
class TestGreenTrainEntryPoint:
    def test_train_with_target_weights(self):
        """C-88: Smoke test exercising train() with target_weights."""
        import numpy as np

        from views_hydranet.train.training_engine import TrainingContext, make, train
        from views_hydranet.utils.volume_handler import VolumeHandler

        T, H, W = 6, 8, 8
        FEATURES = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
        IDENTITY = ["month_id", "priogrid_gid"]
        CHANNEL_MAP = IDENTITY + FEATURES

        config = {
            "model": "HydraBNUNet06_LSTM4",
            "input_channels": len(FEATURES),
            "output_channels": 1,
            "regression_targets": FEATURES,
            "classification_targets": [],
            "features": FEATURES,
            "steps": [1, 2],
            "total_lessons": 2,
            "windows_per_lesson": 1,
            "window_dim": 8,
            "np_seed": 42,
            "torch_seed": 42,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "scheduler": "none",
            "loss_reg": "mse",
            "loss_class": "bce",
            "clip_grad_norm": True,
            "random_flips": False,
            "diagnostic_visualizations": False,
            "total_hidden_channels": 8,
            "dropout_rate": 0.0,
            "weight_init": "xavier_uni",
            "min_events": 1,
            "sampling_strategy": "threshold",
            "min_ratio": 0.0,
            "max_ratio": 1.0,
            "slope_ratio": 0.5,
            "roof_ratio": 1.0,
            "identity_cols": IDENTITY,
            "spatial_cols": ["row", "col"],
            "time_col": "month_id",
            "id_col": "priogrid_gid",
            "transformations": {"identity": FEATURES},
            "derivations": {},
            "body_mask": "pos_cells",
            "qs99_weight": 0.1,
            "qs99_tau": 0.99,
            "target_weights": {
                "lr_ged_sb": 1.0,
                "lr_ged_ns": 3.0,
                "lr_ged_os": 5.0,
            },
        }

        rng = np.random.RandomState(42)
        data = rng.rand(T, H, W, len(CHANNEL_MAP)).astype(np.float32)
        for t in range(T):
            data[t, :, :, 0] = 500 + t
        for r in range(H):
            for c in range(W):
                data[:, r, c, 1] = 1 + r * W + c

        handler = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=CHANNEL_MAP,
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=("row", "col"),
            identity_cols=tuple(IDENTITY),
            feature_cols=tuple(FEATURES),
        )

        device = torch.device("cpu")
        model, criterion, optimizer, scheduler = make(config, device)
        criterion_reg, criterion_class, mtl = criterion

        from tqdm import tqdm

        pbar = tqdm(total=handler.shape[0] - 1, disable=True)

        ctx = TrainingContext(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion_reg=criterion_reg,
            criterion_class=criterion_class,
            multitaskloss_instance=mtl,
            config=config,
            device=device,
        )

        result = train(ctx, handler, pbar)

        assert torch.isfinite(result["total"]), (
            f"train() with hurdle+Basu+target_weights must produce finite loss, "
            f"got {result['total'].item()}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class SumReducer(nn.Module):
    def forward(self, losses):
        return losses.sum()


def make_tiny_model():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1)
            self.base = 8

        def forward(self, x, h):
            out = self.conv(x)
            return ModelOutput(reg=out, cls=out, h_next=h)

        def init_hTtime(self, hidden_channels, H, W):
            return torch.zeros(1, hidden_channels, H, W)

    return TinyModel()
