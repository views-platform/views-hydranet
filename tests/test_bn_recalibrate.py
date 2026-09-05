"""C-184 BN-recalibration fix: post-training recompute of BatchNorm running stats.

The recurrent BN biases its running stats over T-correlated timesteps during training, so eval-mode
over-amplifies and the gate saturates (~40% of seeds — the seed-bimodal eval collapse). The fix
recomputes the running stats forward-only post-training. These tests pin the helper's invariants
(reset + momentum=None + recompute + eval-mode) and the default-on integration flag.
"""

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (  # noqa: E402
    HydraBNUNet06_LSTM4,
)
from views_hydranet.train import training_engine as te  # noqa: E402


def _tiny_model():
    return HydraBNUNet06_LSTM4(
        input_channels=3, total_hidden_channels=8, output_channels=1, dropout_rate=0.0
    )


def test_recalibrate_bn_resets_recomputes_and_evals(monkeypatch):
    """_recalibrate_bn must: set every BN momentum=None, re-accumulate running stats forward-only
    (num_batches_tracked > 0), and leave the model in eval mode — all without touching weights."""
    model = _tiny_model()
    H = W = 16
    x = torch.randn(2, 3, H, W)
    h0 = model.init_hTtime(model.base, H, W)

    # Populate BN stats by a training-mode forward, then snapshot a conv weight (must NOT change).
    model.train()
    model(x, h0)
    w_before = model.enc_conv0.weight.detach().clone()

    # Monkeypatch train() to a forward that updates BN (no real data handler needed).
    # **kw so the stub keeps accepting new keyword arguments as `train` grows; #311 added
    # `apply_input_noise=False` here, which this pass passes precisely so BN statistics are
    # recomputed on CLEAN inputs.
    monkeypatch.setattr(te, "train", lambda ctx, sh, pbar, stage_label="", **kw: model(x, h0))
    ctx = SimpleNamespace(model=model)
    sampler = SimpleNamespace(get_batch=lambda t, th, batch_size=1: ([None], None))
    planner = SimpleNamespace(get_lesson=lambda w: ("lr_sb_best", 0))
    config = {"windows_per_lesson": 1, "bn_recal_windows": 3}

    te._recalibrate_bn(ctx, sampler, planner, config)

    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert bns, "model should have BatchNorm layers"
    assert all(m.momentum is None for m in bns), "BN momentum must be None (cumulative average)"
    assert all(int(m.num_batches_tracked) > 0 for m in bns), "BN stats must be re-accumulated"
    assert not model.training, "model must be left in eval mode after recalibration"
    assert torch.equal(model.enc_conv0.weight, w_before), "recal must NOT change weights"


def test_bn_recalibrate_default_on_in_training_loop():
    """training_loop must call _recalibrate_bn by default (config.get('bn_recalibrate', True)) and
    skip it only for a bn_recal_from experiment run."""
    src = inspect.getsource(te.training_loop)
    assert "_recalibrate_bn(" in src, "training_loop must invoke the BN-recal pass"
    assert 'config.get("bn_recalibrate", True)' in src, "BN-recal must default ON"
    assert "not _bn_recal" in src, "must skip recal for a bn_recal_from-only run"
