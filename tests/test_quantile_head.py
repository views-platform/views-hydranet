"""Red tests (TDD) for the M1 quantile head — the contract M1 must satisfy.

These FAIL until M1 implements `views_hydranet/utils/quantile_head.py` and adds `"quantile"` to the
`output_distribution` validator. Imports are inside each test so a missing module fails the test
(red)
without breaking collection of the rest of the suite. Pre-registration:
reports/2026-07-15_quantile_head_build_dossier/05_analysis_plan.md.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ── (1) config validator accepts the new distribution ────────────────────────
def test_config_accepts_quantile(valid_config_dict):
    """HydraNetConfig must accept output_distribution='quantile' (currently rejected → RED)."""
    from views_hydranet.utils.config_initializer import HydraNetConfig

    cfg = dict(valid_config_dict)
    cfg["output_distribution"] = "quantile"
    config = HydraNetConfig(**cfg)
    assert config.output_distribution == "quantile"


# ── (2) monotone head activation: cumulative softplus → strictly increasing ──
def test_monotone_quantiles_strictly_increasing():
    """`monotone_quantiles(raw)` maps K raw channels → K strictly-increasing quantiles (unbounded).

    base + cumsum(softplus(gaps)); softplus > 0 ⇒ strictly increasing. Unbounded on purpose (alive
    gradient); inference finiteness is the emit clamp's job, not the head's.
    """
    from views_hydranet.utils.quantile_head import monotone_quantiles

    torch.manual_seed(0)
    raw = torch.randn(4, 7, 50)
    q = monotone_quantiles(raw)
    assert q.shape == raw.shape
    assert torch.all(q[..., 1:] - q[..., :-1] > 0), "quantiles must be strictly increasing"
    assert torch.isfinite(q).all()


# ── (3) inverse-CDF resample: quantiles → (..., n_samples), recovers quantiles ─
def test_quantiles_to_samples_shape_and_recovers_quantiles():
    """`quantiles_to_samples(q, taus, n_samples)` = deterministic equiprobable inverse-CDF draws.

    Output shape (..., n_samples), monotone along the sample axis, and its empirical quantiles at
    `taus` recover the input quantiles (the midpoint-grid ensemble identity). numpy carrier — this
    fills
    the existing [T,H,W,C,S] cube.
    """
    from views_hydranet.utils.quantile_head import quantiles_to_samples

    K = 50
    taus = (np.arange(K) + 0.5) / K
    # a monotone quantile function per cell (2 cells): linear then a heavy jump at the top
    base = np.linspace(0.0, 20.0, K)
    q = np.stack([base, base * 2.0 + 3.0])  # (2, K), both strictly increasing
    S = 128
    samples = quantiles_to_samples(q, taus, S)
    assert samples.shape == (2, S)
    assert np.all(np.diff(samples, axis=-1) >= 0), "samples must be sorted (inverse-CDF draws)"
    # empirical quantiles of the draws recover the input quantile function
    recovered = np.quantile(samples, taus, axis=-1).T  # (2, K)
    assert np.allclose(recovered, q, atol=1.0), "resample must recover the input quantiles"


# ── (4) multi-τ pinball loss: differentiable + minimised at the true quantiles ─
def test_quantile_loss_differentiable_and_minimised_at_truth():
    """`QuantileLoss(taus)(pred_quantiles, target)` = multi-τ pinball; lower when the
    predicted quantiles match the target distribution's quantiles than when they're shifted."""
    from views_hydranet.utils.quantile_head import QuantileLoss

    K = 50
    taus = torch.tensor((np.arange(K) + 0.5) / K, dtype=torch.float32)
    loss_fn = QuantileLoss(taus)

    torch.manual_seed(0)
    target = torch.distributions.LogNormal(1.0, 1.0).sample((2000,))  # heavy-ish positive target
    true_q = torch.quantile(target, taus).expand(2000, K).clone().requires_grad_(True)

    good = loss_fn(true_q, target)
    good.backward()
    assert true_q.grad is not None and torch.isfinite(good), "loss must be finite + differentiable"

    shifted = true_q.detach() + 5.0  # miscalibrated quantiles
    assert loss_fn(true_q.detach(), target) < loss_fn(shifted, target), (
        "pinball must be lower at the true quantiles than at shifted ones"
    )


# ── (5) end-to-end wiring (CPU): quantile model forward + loss + backward ─────
def test_model_quantile_head_forward_and_train_step():
    """The quantile head forwards to [B, 3K, H, W] with each target's K channels strictly monotone,
    no dead-fan explosion, and choose_loss('quantile') gives a finite differentiable pinball."""
    from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
    from views_hydranet.utils.quantile_head import QuantileLoss
    from views_hydranet.utils.utils import choose_loss

    H = W = 16
    in_ch, hid_ch, k = 8, 32, 50
    torch.manual_seed(0)
    m = (
        HydraBNUNet06_LSTM4(in_ch, hid_ch, 1, 0.0, output_distribution="quantile", n_quantiles=k)
        .float()
        .train()
    )
    x = torch.randn(1, in_ch, H, W)
    h = m.init_hTtime(hid_ch, H, W).float()
    out = m(x, h)

    assert out.reg.shape == (1, 3 * k, H, W), "3 reg heads × K quantiles"
    for j in range(3):  # each target's K channels strictly increasing along the channel axis
        block = out.reg[:, j * k : (j + 1) * k]
        assert torch.all(block[:, 1:] - block[:, :-1] > 0), f"target {j} quantiles not increasing"
    assert torch.isfinite(out.reg).all(), "finite head output"

    cfg = {
        "loss_reg": "quantile",
        "loss_class": "bce",
        "n_quantiles": k,
        "regression_targets": ["a", "b", "c"],
        "classification_targets": ["a", "b", "c"],
    }
    crit_reg, _, _ = choose_loss(cfg, torch.device("cpu"))
    assert isinstance(crit_reg, QuantileLoss)
    target = torch.rand(1, H, W) * 5.0
    pred_q = out.reg[:, 0:k].permute(0, 2, 3, 1)  # target 0's K quantiles → [B,H,W,K]
    loss = crit_reg(pred_q, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
