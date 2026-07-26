"""H-SAMPLE / EXP-2 (bloom dossier): the `rollout_feedback` config flag.

`rollout_feedback='mean'` (default) feeds back the emit-mean E[y] in the autoregressive loop — the
historical behavior (byte-identical). `rollout_feedback='sample'` feeds back a single seeded family
draw per cell (the ancestral rollout). Only the fed-back copy changes; the scored cube still comes
from the recorded params. Fail-loud on a bad value or 'sample' without a registered family.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
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


# ── fail-loud validation ─────────────────────────────────────────────────


def test_rollout_feedback_rejects_bad_value():
    with pytest.raises(ValueError, match="rollout_feedback"):
        _make_inf("nb", rollout_feedback="bogus")


def test_rollout_feedback_sample_requires_family():
    # 'standard' (legacy) has no registered family → 'sample' must fail loud.
    with pytest.raises(ValueError, match="registered distribution family"):
        _make_inf("standard", rollout_feedback="sample")


# ── parity: default / 'mean' is byte-identical ───────────────────────────


def test_rollout_feedback_default_equals_mean_byte_identical():
    h = _mock_handler()
    mag_default, _ = _make_inf("nb").generate_posterior_samples(h, origin=1)
    mag_mean, _ = _make_inf("nb", rollout_feedback="mean").generate_posterior_samples(h, origin=1)
    assert np.array_equal(mag_default, mag_mean)  # unset == 'mean', exactly


# ── 'sample' changes the trajectory, and is deterministic ────────────────


def test_rollout_feedback_sample_differs_from_mean():
    h = _mock_handler()
    mag_mean, _ = _make_inf("nb", rollout_feedback="mean").generate_posterior_samples(h, origin=1)
    inf_s = _make_inf("nb", rollout_feedback="sample")
    mag_samp, _ = inf_s.generate_posterior_samples(h, origin=1)
    # seed step has no feedback yet → identical there; later horizons diverge.
    assert not np.array_equal(mag_mean, mag_samp)


def test_rollout_feedback_sample_is_deterministic():
    h = _mock_handler()
    a, _ = _make_inf("nb", rollout_feedback="sample").generate_posterior_samples(h, origin=1)
    b, _ = _make_inf("nb", rollout_feedback="sample").generate_posterior_samples(h, origin=1)
    assert np.array_equal(a, b)  # seeded generator (torch_seed + pass idx) → reproducible


# ── EXP-3 oracle: teacher_forced feeds real input, differs from mean/sample ───


def test_rollout_feedback_teacher_forced_differs_and_deterministic():
    h = _mock_handler()
    mag_mean, _ = _make_inf("nb", rollout_feedback="mean").generate_posterior_samples(h, origin=1)
    tf1 = _make_inf("nb", rollout_feedback="teacher_forced")
    tf2 = _make_inf("nb", rollout_feedback="teacher_forced")
    o1, _ = tf1.generate_posterior_samples(h, origin=1)
    o2, _ = tf2.generate_posterior_samples(h, origin=1)
    assert not np.array_equal(mag_mean, o1)  # real-input feedback changes the trajectory
    assert np.array_equal(o1, o2)  # deterministic (no extra randomness)


def test_rollout_feedback_sample_composes_with_gate_soft():
    """nb + soft_gate: sample-feedback composes the draw with the gate (not the ungated native
    draw), so the mean/sample A/B isolates feedback content, not gated-vs-ungated."""
    h = _mock_handler()
    inf = _make_inf("nb", rollout_feedback="sample")
    inf.config["forecast_composition"] = "soft_gate"
    mag_samp, _ = inf.generate_posterior_samples(h, origin=1)
    inf_m = _make_inf("nb", rollout_feedback="mean")
    inf_m.config["forecast_composition"] = "soft_gate"
    mag_mean, _ = inf_m.generate_posterior_samples(h, origin=1)
    assert not np.array_equal(mag_mean, mag_samp)  # composed sample-feedback changes the path
