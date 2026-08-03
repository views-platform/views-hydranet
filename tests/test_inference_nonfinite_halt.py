"""Behavioral RED tests for the inference-time non-finite HALT guard (ADR-003: Fail Loud).

``HydraNetInference.predict`` (and thus ``generate_posterior_samples``) accumulates the emitted
magnitudes/params across the rollout and, before returning, asserts they are all finite —
raising ``RuntimeError`` if the model ever emitted a NaN/Inf
(hydranet_inference.py:610-616 for magnitudes, :593-599 for params).

These tests assert the BEHAVIOR — that a NaN emitted by the model actually HALTS inference —
rather than grepping the source for a ``raise`` statement (the tautological check they replace).
We corrupt the model two ways: (1) wrap ``forward`` so ``output.reg`` carries a NaN, and
(2) set a real weight to NaN via ``torch.no_grad()``. Both must trip the same fail-loud guard.

Setup mirrors tests/distributions/test_rollout_feedback.py (`_make_inf` + `_mock_handler`).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (  # noqa: E402
    HydraBNUNet06_LSTM4,
)
from views_hydranet.utils.hydranet_inference import HydraNetInference  # noqa: E402

_FEATURES = ["lr_sb", "lr_ns", "lr_os", "by_sb", "by_ns", "by_os"]


def _make_inf(output_distribution, *, rollout_feedback=None, time_steps=3):
    torch.manual_seed(0)
    model = HydraBNUNet06_LSTM4(3, 16, 1, 0.0, output_distribution=output_distribution).float()
    config = {
        "steps": list(range(1, time_steps + 1)),
        "time_steps": time_steps,
        "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
        "classification_targets": ["by_sb", "by_ns", "by_os"],
        "features": ["lr_sb", "lr_ns", "lr_os"],
        "static_channels": [],
        "n_posterior_samples": 2,
        "n_head_samples": 1,
        "np_seed": 42,
        "torch_seed": 1234,
    }
    if rollout_feedback is not None:
        config["rollout_feedback"] = rollout_feedback
    return HydraNetInference(model, config, device="cpu")


def _mock_handler(seq_len=6, h=8, w=8):
    tensor = torch.rand(1, seq_len, len(_FEATURES), h, w).abs()

    def to_pytorch(device, include_identities=False):
        return tensor.to(device)

    return SimpleNamespace(
        to_pytorch=to_pytorch,
        channel_map=list(_FEATURES),
        feature_cols=list(_FEATURES),
        time_col="month_id",
        data=None,
    )


# ── the guard actually HALTS on a non-finite emission ─────────────────────


def test_nonfinite_reg_head_halts_inference():
    """Model emits a NaN in ``output.reg`` -> ``generate_posterior_samples`` raises RuntimeError.

    We wrap ``forward`` (architecture-agnostic) so every call injects a NaN into the activated
    regression params; that NaN flows through ``_emit_magnitude`` into the accumulated magnitudes,
    which the pre-return finiteness check must reject (fail-loud), rather than returning a
    NaN-poisoned cube.

    We use ``rollout_feedback='mean'`` (the emit-mean path) so the NaN reaches the emitted-
    magnitude finiteness guard directly, rather than the family sampler tripping its own
    Poisson-rate check first — this test targets the inference HALT guard specifically."""
    inf = _make_inf("nb", rollout_feedback="mean")
    model = inf.model
    orig_forward = model.forward

    def nan_forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        reg = out.reg.clone()
        reg[..., 0, 0] = float("nan")  # poison one cell of the emitted params
        return out._replace(reg=reg)

    model.forward = nan_forward

    with pytest.raises(RuntimeError, match="non-finite"):
        inf.generate_posterior_samples(_mock_handler(), origin=1)


def test_nonfinite_weight_halts_inference():
    """A NaN WEIGHT (corrupted via ``torch.no_grad()``) propagates to a non-finite emission and the
    guard HALTS — the honest behavioral analogue of a real numerical explosion, not a source grep.

    We NaN the entire final regression-head conv weight so the corruption is guaranteed to reach
    ``output.reg`` regardless of the input, then assert inference fails loud. ``rollout_feedback=
    'mean'`` routes the NaN to the magnitude finiteness guard (not the family sampler)."""
    inf = _make_inf("nb", rollout_feedback="mean")
    model = inf.model

    # Corrupt the last Conv2d weight in the module tree (the regression emission head sits at the
    # very end of the U-Net), guaranteeing a non-finite reg output.
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    assert last_conv is not None, "expected at least one Conv2d in the model"
    with torch.no_grad():
        last_conv.weight.fill_(float("nan"))

    with pytest.raises(RuntimeError, match="non-finite"):
        inf.generate_posterior_samples(_mock_handler(), origin=1)


def test_nonfinite_params_return_path_halts():
    """The ``return_params=True`` branch has its OWN finiteness guard (:593-599). A NaN in the
    activated params must halt that path too, so upstream family samplers never receive a poisoned
    param cube. ``rollout_feedback='mean'`` keeps the family sampler out of the path so the params
    guard is the one that fires."""
    inf = _make_inf("nb", rollout_feedback="mean")
    model = inf.model
    orig_forward = model.forward

    def nan_forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        reg = out.reg.clone()
        reg[..., 0, 0] = float("nan")
        return out._replace(reg=reg)

    model.forward = nan_forward

    t = _mock_handler().to_pytorch("cpu")
    with pytest.raises(RuntimeError, match="non-finite"):
        inf.predict(t, origin=1, sample_idx=0, feature_names=_FEATURES, return_params=True)
