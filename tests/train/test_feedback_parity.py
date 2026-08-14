"""Train↔inference scheduled-sampling feedback PARITY (C-246 / C-259 — the missing anchor).

Enabling scheduled sampling (ss_epsilon_max>0) makes the TRAINING feedback
(`training_engine._family_feedback_log1p`) run against inference's rollout feedback
(`hydranet_inference._sample_feedback`) for the first time in a real run. The exposure-bias premise
is only meaningful if the two construct the SAME object. This pins that: for
`truncated_nb`/`nb` + `soft_gate` + `emit_family_core=False`, with a shared seeded generator, the
two feedbacks must be BYTE-EQUAL (this test runs on CPU; production byte-equality holds only on
the SAME device — on GPU the CUDA vs CPU generator streams differ, so parity is then
distributional, not byte-exact). A divergence here silently invalidates any SS verdict.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.distributions import resolve_family  # noqa: E402
from views_hydranet.train.training_engine import _family_feedback_log1p  # noqa: E402


def _make_inference(output_distribution, composition, *, emit_family_core=False):
    """A HydraNetInference for the given family + composition (only _family + config are read by
    _sample_feedback — the model forward is not exercised)."""
    from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
    from views_hydranet.utils.hydranet_inference import HydraNetInference

    torch.manual_seed(0)
    model = HydraBNUNet06_LSTM4(3, 16, 1, 0.0, output_distribution=output_distribution).float()
    config = {
        "steps": [1],
        "time_steps": 1,
        "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
        "classification_targets": ["by_sb", "by_ns", "by_os"],
        "features": ["lr_sb", "lr_ns", "lr_os"],
        "static_channels": [],
        "n_posterior_samples": 1,
        "n_head_samples": 1,
        "np_seed": 42,
        "torch_seed": 1234,
        "forecast_composition": composition,
        "emit_family_core": emit_family_core,
        "gate_threshold": None,
        "rollout_feedback": "sample",
    }
    return HydraNetInference(model, config, device="cpu")


def _reg_and_gate(fam, b=2, h=6, w=6, n_reg=3, seed=0):
    """Activated params [b, n_reg*npar, h, w] + a sigmoid gate [b, n_reg, h, w]."""
    g = torch.Generator().manual_seed(seed)
    npar = fam.n_params

    def _one():
        return fam.activate(torch.randn(b, h, w, npar, generator=g)).permute(0, 3, 1, 2)

    cols = [_one() for _ in range(n_reg)]
    reg = torch.cat(cols, dim=1)
    prob = torch.sigmoid(torch.randn(b, n_reg, h, w, generator=g))
    return reg, prob


@pytest.mark.parametrize("fam_name", ["truncated_nb", "nb"])
def test_train_infer_feedback_byte_parity_sample_soft_gate(fam_name):
    """THE anchor: with a shared seeded generator, training feedback == inference feedback."""
    fam = resolve_family(fam_name)
    reg, prob = _reg_and_gate(fam)
    inf = _make_inference(fam_name, "soft_gate")

    train_fb = _family_feedback_log1p(
        reg, fam, "sample", prob, "soft_gate", None, torch.Generator().manual_seed(123)
    )
    infer_fb = inf._sample_feedback(reg, prob, torch.Generator().manual_seed(123))

    assert train_fb.shape == infer_fb.shape == (2, 3, 6, 6)
    assert torch.equal(train_fb, infer_fb), (
        f"[{fam_name}] train vs inference SS feedback DIVERGED — exposure mismatch would "
        "silently invalidate any scheduled-sampling verdict"
    )


@pytest.mark.parametrize("fam_name", ["truncated_nb", "nb"])
def test_train_infer_feedback_byte_parity_self_zeroed(fam_name):
    """self_zeroed composition (no gate): the two feedbacks must still match byte-for-byte."""
    fam = resolve_family(fam_name)
    reg, prob = _reg_and_gate(fam)
    inf = _make_inference(fam_name, "self_zeroed")
    train_fb = _family_feedback_log1p(
        reg, fam, "sample", prob, "self_zeroed", None, torch.Generator().manual_seed(7)
    )
    infer_fb = inf._sample_feedback(reg, prob, torch.Generator().manual_seed(7))
    assert torch.equal(train_fb, infer_fb)


def test_training_feedback_is_reproducible_under_generator():
    """C-261: a seeded generator makes the training SS draw byte-reproducible (was global RNG)."""
    fam = resolve_family("truncated_nb")
    reg, prob = _reg_and_gate(fam)
    a = _family_feedback_log1p(
        reg, fam, "sample", prob, "soft_gate", None, torch.Generator().manual_seed(1)
    )
    b = _family_feedback_log1p(
        reg, fam, "sample", prob, "soft_gate", None, torch.Generator().manual_seed(1)
    )
    assert torch.equal(a, b)
