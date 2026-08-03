"""S2 / Epic #193 — regression guard for the sample-feedback bloom fix's INVARIANT.

The C-113 bloom is a property of a **trained** model's out-of-distribution io-gain (EXP-2/EXP-3):
the dense emit-mean fed back each step is OOD vs sparse conflict, pushing gain > 1. A fast/CPU test
on an untrained model therefore CANNOT reproduce "mean blooms / sample bounded" without engineering
the model to fake it — the **end-to-end bloom-bounded proof is owned by S6 (#199, trained eval)**.

What a fast guard CAN protect is the fix's DEFINING INVARIANT: `rollout_feedback='sample'` feeds
back a **sparse, in-distribution field** (a composition-aware family draw), NOT the dense emit-mean
(the bloom driver). If `_sample_feedback` regresses to feeding the mean / raw params, the sparsity
collapses and this guard fails.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (  # noqa: E402
    HydraBNUNet06_LSTM4,
)
from views_hydranet.distributions import resolve_family  # noqa: E402
from views_hydranet.utils.hydranet_inference import HydraNetInference  # noqa: E402

_FEATURES = ["lr_sb", "lr_ns", "lr_os", "by_sb", "by_ns", "by_os"]

# (output_distribution, forecast_composition) — the 3 valid deployable arms (ADR-069).
_ARMS = [
    ("nb", "soft_gate"),  # gated_NB
    ("nb", "threshold_gate"),  # th_gated_NB
    ("zinb", "self_zeroed"),  # ZINB
]


def _make_inf(output_distribution, composition, *, rollout_feedback="sample", time_steps=3):
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
        "forecast_composition": composition,
        "rollout_feedback": rollout_feedback,
    }
    if composition == "threshold_gate":
        config["gate_threshold"] = 0.5
    return HydraNetInference(model, config, device="cpu")


def _sparse_params(fam, comp, *, b=1, h=8, w=8, n_reg=3):
    """Activated params that yield a SPARSE draw: small mu (softplus of a negative raw) so NB draws
    are mostly 0. The gate is composition-aware so the sample-vs-mean sparsity gap is visible:
    soft_gate → low gate (Bernoulli drops most draws); threshold_gate → gate ABOVE τ (kept, so the
    gap comes from the integer draw of a small mu being mostly 0 vs the nonzero mean); self_zeroed
    → gate unused (zinb self-zeroes via structural pi). Returns (reg, gate)."""
    npar = fam.n_params
    g = torch.Generator().manual_seed(0)
    cols = []
    for _ in range(n_reg):
        raw = torch.full((b, h, w, npar), -3.0) + 0.01 * torch.randn(b, h, w, npar, generator=g)
        cols.append(fam.activate(raw).permute(0, 3, 1, 2))
    reg = torch.cat(cols, dim=1)
    gate_val = {"soft_gate": 0.1, "threshold_gate": 0.8, "self_zeroed": 0.5}[comp]
    gate = torch.full((b, n_reg, h, w), gate_val)
    return reg, gate


def _zero_fraction(t):
    return float((t <= 1e-8).float().mean())


# ── the invariant: sample-feedback is SPARSE, the mean field is DENSE ─────────


@pytest.mark.parametrize("dist,comp", _ARMS)
def test_sample_feedback_field_is_sparser_than_mean(dist, comp):
    """The fix's defining property: the composed sample field has far more zeros than the composed
    mean field for the same params. Regressing to feed the dense mean/raw fails this."""
    inf = _make_inf(dist, comp)
    fam = resolve_family(dist)
    reg, gate = _sparse_params(fam, comp)
    gen = torch.Generator(device="cpu").manual_seed(7)
    sample_fb = inf._sample_feedback(reg, gate, gen)  # composed draw, log1p [B, n_reg, H, W]
    mean_fb = inf._emit_magnitude(reg, gate)  # composed mean, log1p [B, n_reg, H, W]
    assert torch.isfinite(sample_fb).all() and torch.isfinite(mean_fb).all()
    zf_sample, zf_mean = _zero_fraction(sample_fb), _zero_fraction(mean_fb)
    assert zf_sample > zf_mean + 0.2, (
        f"[{dist}+{comp}] sample-feedback not materially sparser than mean "
        f"(zeros sample={zf_sample:.2f} vs mean={zf_mean:.2f}) — sparse-field invariant broken"
    )


@pytest.mark.parametrize("dist,comp", _ARMS)
def test_sample_feedback_is_deterministic(dist, comp):
    inf = _make_inf(dist, comp)
    fam = resolve_family(dist)
    reg, gate = _sparse_params(fam, comp)
    a = inf._sample_feedback(reg, gate, torch.Generator(device="cpu").manual_seed(7))
    b = inf._sample_feedback(reg, gate, torch.Generator(device="cpu").manual_seed(7))
    assert torch.equal(a, b)


# ── integration plumbing: multi-step mean vs sample rollout is finite + differs ─


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


@pytest.mark.parametrize("dist,comp", _ARMS)
def test_multistep_mean_vs_sample_finite_and_differs(dist, comp):
    """End-to-end plumbing through generate_posterior_samples over a multi-step horizon: both modes
    stay finite; sample differs from mean at h>=2. (Trained-model bloom-bounded proof is S6.)"""
    hdl = _mock_handler()
    mean_inf = _make_inf(dist, comp, rollout_feedback="mean")
    samp_inf = _make_inf(dist, comp, rollout_feedback="sample")
    mag_mean, _ = mean_inf.generate_posterior_samples(hdl, origin=1)
    mag_samp, _ = samp_inf.generate_posterior_samples(hdl, origin=1)
    assert np.isfinite(mag_mean).all() and np.isfinite(mag_samp).all()
    assert not np.array_equal(mag_mean, mag_samp)  # the flag is live end-to-end
