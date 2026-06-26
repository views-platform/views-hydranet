"""Falsification stub — dead-ReLU regression body (2026-06-25).

FINDING (HARD): for the hurdle point/shrinkage family the regression head uses ReLU
(HydraBNrecurrentUnet_06_LSTM4.py:85 — softplus ONLY for hurdle_nb, else relu). Under the
active_window mask (heavy zero supervision) the rare targets' pre-activation drifts fully
negative, so ReLU clamps the body to identically 0 with zero gradient — an unrecoverable
DEAD head. Empirically (aw seed-11 artifact, real forward): lr_ns_best and lr_os_best emit
out_reg == 0 on 100% of cells incl. event cells (pre-activation max < 0), while the gate fires
normally. This is the mechanism behind the #66 ns/os flatline and the #73 shrinkage puzzle
(no loss can resurrect a zero-gradient ReLU).

These tests encode the defect. They FAIL against current code by design. The fix direction is a
DECISION (e.g. softplus body for the hurdle point/shrinkage compose, which can't die), so the
assertion below is the proposed contract, not yet ratified.
"""
import pytest

torch = pytest.importorskip("torch")


def _build(output_distribution, reg_activation=None):
    from views_hydranet.utils.utils import choose_model
    cfg = {
        "model": "HydraBNUNet06_LSTM4",
        "input_channels": 3,
        "output_channels": 1,
        "total_hidden_channels": 32,
        "dropout_rate": 0.0,
        "weight_init": "xavier_norm",
        "h_init": "abs_rand_exp-100",
        "output_distribution": output_distribution,
    }
    if reg_activation is not None:
        cfg["reg_activation"] = reg_activation
    return choose_model(cfg, torch.device("cpu"))


def test_hurdle_shrinkage_body_uses_nondying_activation():
    """A ReLU body under hurdle masking can die (zero output + zero gradient, unrecoverable).
    The hurdle point/shrinkage head should use a non-dying activation (softplus) so a rare
    target cannot collapse to identically-zero. FAILS today (relu)."""
    model = _build("hurdle_shrinkage")
    name = getattr(model._reg_activation, "__name__", str(model._reg_activation))
    assert name == "softplus", (
        f"hurdle_shrinkage reg head uses '{name}' — a hard-zero ReLU that dies on rare "
        f"targets under the active_window mask (ns/os emit identically 0). Use softplus."
    )


def test_relu_body_can_emit_identically_zero_with_zero_gradient():
    """Characterizes the dead-ReLU trap directly: a strongly-negative pre-activation yields
    out=0 AND grad=0, so the loss cannot recover it. Documents why no body loss helped ns/os."""
    import torch.nn.functional as F

    h_reg = torch.full((4, 4), -5.0, requires_grad=True)  # learned dead state (pre-activation < 0)
    out = F.relu(h_reg)
    target = torch.full((4, 4), 2.94)  # log1p(~18): a real event magnitude
    loss = ((out - target) ** 2).mean()
    loss.backward()
    # The defect: output is identically 0 and NO gradient flows back to revive it.
    assert out.max().item() == 0.0
    # zero gradient => unrecoverable; this is the bug
    assert h_reg.grad.abs().max().item() == 0.0
