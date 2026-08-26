"""Shared harness for the training-loop gradient tests.

Two vehicles, because the two things under test live at different levels:

* :func:`seq_fixture` — the real ``HydraBNUNet06_LSTM4`` plus a sparse log1p field, for anything
  that needs ``_process_sequence`` and a genuine autograd graph (BPTT reach, per-parameter
  gradient health). ``TinyModel`` in the top-level ``tests/conftest.py`` has neither BatchNorm nor
  a ConvLSTM, so it cannot exercise either.
* :func:`loop_config` / :func:`loop_handler` — a minimal but *real* end-to-end ``training_loop``
  vehicle (real ``make()``, real optimizer, real data), for the guards that are inline in
  ``training_loop`` and therefore unreachable any other way: the gradient clip and the
  ``if w_loss > 0`` backward gate.

Both were previously private helpers duplicated across test modules
(``tests/train/test_pushforward.py::_fixture``, ``tests/test_optimization_gate.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from views_hydranet.architectures.registry import get_architecture  # noqa: E402
from views_hydranet.distributions import get_family  # noqa: E402
from views_hydranet.train.training_engine import _SequenceIndices  # noqa: E402
from views_hydranet.utils.volume_handler import VolumeHandler  # noqa: E402

#: Production target names, so the channel-role accessors resolve the same way they do in a run.
FEATS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
CLS = ["by_sb_best", "by_ns_best", "by_os_best"]


class SumLoss(nn.Module):
    """Stand-in for ``MultiTaskLoss``: a plain sum, so a test reads the real per-task terms.

    Deliberately NOT the real balancer — tests that care about the balancer construct it directly.
    """

    def forward(self, losses):  # noqa: D102
        return losses.sum()


def make_sequence(*, seed: int = 0, T: int = 6, hw: int = 8, trained_ckpt: str | None = None):
    """Return ``(x, model, h0, idx, family)`` on the real architecture.

    ``x`` is a ``[1, T, 6, hw, hw]`` log1p-space field with ~3% active cells — sparse like the real
    grid, because a dense field puts the NB head in a regime it never sees in training.
    """
    torch.manual_seed(seed)
    cfg = {
        "features": FEATS,
        "regression_targets": FEATS,
        "classification_targets": CLS,
        "static_channels": [],
    }
    idx = _SequenceIndices(FEATS + CLS, cfg)
    model = get_architecture("HydraBNUNet06_LSTM4")(
        len(FEATS), 32, 1, 0.0, output_distribution="nb"
    ).float()
    if trained_ckpt is not None:
        model.load_state_dict(torch.load(trained_ckpt, map_location="cpu", weights_only=False))
    model.train()
    h = model.init_hTtime(model.base, hw, hw).float()
    n = len(FEATS) + len(CLS)
    x = (torch.rand(1, T, n, hw, hw) < 0.03).float() * torch.rand(1, T, n, hw, hw) * 4
    return x, model, h, idx, get_family("nb")


@pytest.fixture
def seq_fixture():
    """Factory for :func:`make_sequence`, so a test can pick its own seed / length."""
    return make_sequence


def loop_config(**overrides):
    """A minimal config that drives the REAL ``training_loop`` end to end.

    Small enough to run in seconds, real enough that ``make()`` builds the production
    architecture, optimizer and losses — which is the only way to reach the guards that are
    inline in ``training_loop``.
    """
    cfg = {
        "model": "HydraBNUNet06_LSTM4",
        "base": 16,
        "input_channels": 6,
        "total_hidden_channels": 16,
        "output_channels": 6,
        "dropout_rate": 0.1,
        "regression_targets": ["lr_f1", "lr_f2", "lr_f3"],
        "classification_targets": ["by_f1", "by_f2", "by_f3"],
        "total_lessons": 2,
        "windows_per_lesson": 1,
        "steps": [1],
        "time_steps": 1,
        "n_posterior_samples": 1,
        "np_seed": 42,
        "torch_seed": 42,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "window_dim": 8,
        "weight_init": "xavier_uni",
        "clip_grad_norm": False,
        "time_col": "t",
        "id_col": "i",
        "spatial_cols": ["y", "x"],
        "identity_cols": ["t", "i"],
        "features": ["lr_f1", "lr_f2", "lr_f3", "by_f1", "by_f2", "by_f3"],
        "row_offset": 0,
        "col_offset": 0,
        "height": 8,
        "width": 8,
        "min_events": 0,
        "max_events": 100,
        "sampling_strategy": "threshold",
        "slope_ratio": 1.0,
        "roof_ratio": 1.0,
        "min_ratio": 0.1,
        "max_ratio": 0.9,
        "run_type": "train",
        "scheduler": "none",
        "loss_reg": "mse",
        "loss_class": "focal",
        "loss_reg_a": 1.0,
        "loss_reg_c": 1.0,
        "loss_class_alpha": 0.25,
        "loss_class_gamma": 2.0,
        "transformations": {"identity": ["lr_f1", "lr_f2", "lr_f3"]},
        "derivations": {
            "binary": [
                {"from": "lr_f1", "to": "by_f1", "threshold": 0},
                {"from": "lr_f2", "to": "by_f2", "threshold": 0},
                {"from": "lr_f3", "to": "by_f3", "threshold": 0},
            ]
        },
    }
    cfg.update(overrides)
    return cfg


def loop_handler(cfg, *, seed: int = 0):
    """A ``VolumeHandler`` carrying enough signal that the losses are non-trivial."""
    rng = np.random.default_rng(seed)
    data = rng.random((5, 8, 8, 4))
    data[..., 1:] = 1.0
    return VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=["t", "lr_f1", "lr_f2", "lr_f3"],
        time_col="t",
        id_col="i",
        spatial_cols=["y", "x"],
        identity_cols=["t", "i"],
        feature_cols=["lr_f1", "lr_f2", "lr_f3"],
        config=cfg,
    )


def grad_norm(params) -> float:
    """L2 norm of the concatenated gradient over ``params`` (skipping ``None``)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total**0.5
