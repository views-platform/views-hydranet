"""A-S8 (#175): inference emit-mean + D×K posterior sampler for the families (ADR-067).

`_emit_magnitude` delegates a registered family to `family.mean` → `log1p(E[y])` (= AR feedback);
`generate_posterior_samples` keeps the D MC-dropout passes and draws K = `n_head_samples` per-cell
samples/pass via `family.sample` with a seeded `torch.Generator`, filling the `[T,H,W,C,S]` cube
with S = D×K through one `posterior_S`. A RAM preflight (`assert_cube_fits`) rejects an oversize
cube before allocation. Legacy `output_distribution` values keep their exact path (byte-identical).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from views_hydranet.distributions import resolve_family  # noqa: E402
from views_hydranet.distributions.sampling import to_cube_samples  # noqa: E402

# ── helpers ──────────────────────────────────────────────────────────────


def _activated_params(fam, t, h, w, n_reg, *, seed=0):
    """Build an activated param stack [t, n_reg*n_params, h, w] for a family."""
    g = torch.Generator().manual_seed(seed)
    npar = fam.n_params
    per_target = []
    for _ in range(n_reg):
        raw = torch.randn(t, h, w, npar, generator=g)
        act = fam.activate(raw)  # [t,h,w,npar]
        per_target.append(act.permute(0, 3, 1, 2))  # [t,npar,h,w]
    return torch.cat(per_target, dim=1)  # [t, n_reg*npar, h, w]


def _make_inference(output_distribution, *, n_posterior_samples=2, n_head_samples=1):
    """Build a HydraNetInference around a tiny real model for the given distribution."""
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
        "n_posterior_samples": n_posterior_samples,
        "n_head_samples": n_head_samples,
        "np_seed": 42,
        "torch_seed": 1234,
    }
    return HydraNetInference(model, config, device="cpu")


def _mock_handler(feature_names, *, seq_len=2, h=8, w=8):
    """A minimal handler exposing exactly what generate_posterior_samples reads."""
    b, c = 1, len(feature_names)
    tensor = torch.rand(b, seq_len, c, h, w).abs()

    def to_pytorch(device, include_identities=False):
        return tensor.to(device)

    return SimpleNamespace(
        to_pytorch=to_pytorch,
        channel_map=list(feature_names),
        feature_cols=list(feature_names),
        time_col="month_id",
        data=None,
    )


_FEATURES = ["lr_sb", "lr_ns", "lr_os", "by_sb", "by_ns", "by_os"]


# ── to_cube_samples unit ─────────────────────────────────────────────────


def test_to_cube_samples_shape_and_log1p_space():
    fam = resolve_family("nb")
    t, h, w, n_reg, k = 2, 4, 4, 3, 5
    params = _activated_params(fam, t, h, w, n_reg)
    g = torch.Generator().manual_seed(7)
    cube = to_cube_samples(params, fam, k, g, n_reg)
    assert cube.shape == (t, h, w, n_reg, k)
    assert np.isfinite(cube).all()
    assert (cube >= 0).all()  # log1p of non-negative counts


def test_to_cube_samples_is_generator_deterministic():
    fam = resolve_family("zinb")
    params = _activated_params(fam, 2, 4, 4, 3)
    a = to_cube_samples(params, fam, 4, torch.Generator().manual_seed(11), 3)
    b = to_cube_samples(params, fam, 4, torch.Generator().manual_seed(11), 3)
    c = to_cube_samples(params, fam, 4, torch.Generator().manual_seed(99), 3)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# ── RAM preflight ────────────────────────────────────────────────────────


def test_assert_cube_fits_raises_over_ram(monkeypatch):
    from views_hydranet.utils import disk_guard

    monkeypatch.setattr(disk_guard, "_available_ram_bytes", lambda: 1_000_000)  # 1 MB
    with pytest.raises(RuntimeError, match="RAM|memory|cube"):
        disk_guard.assert_cube_fits((100, 100, 100, 3, 100), dtype=np.float32)


def test_assert_cube_fits_raises_over_budget(monkeypatch):
    from views_hydranet.utils import disk_guard

    monkeypatch.setattr(disk_guard, "_available_ram_bytes", lambda: 10**12)  # plenty
    # ~0.48 GB cube vs a 0.1 GB budget -> raise
    with pytest.raises(RuntimeError, match="budget|cube"):
        disk_guard.assert_cube_fits((100, 100, 100, 3, 40), dtype=np.float32, budget_gb=0.1)


def test_assert_cube_fits_passes_small(monkeypatch):
    from views_hydranet.utils import disk_guard

    monkeypatch.setattr(disk_guard, "_available_ram_bytes", lambda: 10**12)
    disk_guard.assert_cube_fits((2, 8, 8, 3, 6), dtype=np.float32)  # no raise
    disk_guard.assert_cube_fits((2, 8, 8, 3, 6), dtype=np.float32, budget_gb=1.0)


# ── family emit (byte-identical legacy) ──────────────────────────────────


def test_emit_magnitude_family_returns_log1p_mean():
    inf = _make_inference("nb")
    fam = resolve_family("nb")
    b, h, w, n_reg = 1, 8, 8, 3
    params = _activated_params(fam, b, h, w, n_reg)  # [b, n_reg*npar, h, w]
    prob = torch.rand(b, n_reg, h, w)
    emitted = inf._emit_magnitude(params, prob)
    assert emitted.shape == (b, n_reg, h, w)
    # reconstruct expected log1p(mean) target-by-target
    npar = fam.n_params
    for j in range(n_reg):
        pj = params[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1)
        expected = torch.log1p(fam.mean(pj))
        assert torch.allclose(emitted[:, j], expected, atol=1e-6)


def test_emit_magnitude_legacy_stub_unchanged():
    # a SimpleNamespace with no _family (mirrors the A-S2 parity stub) -> legacy identity path
    from views_hydranet.utils.hydranet_inference import HydraNetInference

    stub = SimpleNamespace(output_distribution="standard")
    reg = torch.randn(1, 3, 8, 8)
    out = HydraNetInference._emit_magnitude(stub, reg, torch.rand(1, 3, 8, 8))
    assert torch.equal(out, reg)  # standard = identity


# ── predict(return_params=) ──────────────────────────────────────────────


def test_predict_return_params_shape_and_tuple_preserved():
    inf = _make_inference("nb")
    fam = resolve_family("nb")
    b, seq_len, c, h, w = 1, 2, 6, 8, 8
    full = torch.rand(b, seq_len, c, h, w).abs()

    # default: 2-tuple, emit-magnitude [T, n_reg, H, W]
    mag, prob = inf.predict(full, origin=1, sample_idx=0, feature_names=_FEATURES)
    assert mag.shape[1] == 3  # n_reg emit channels

    # return_params=True: params stack [T, n_reg*n_params, H, W]
    params, prob2 = inf.predict(
        full, origin=1, sample_idx=0, feature_names=_FEATURES, return_params=True
    )
    assert params.shape[1] == 3 * fam.n_params


# ── D×K sampler + determinism + legacy ───────────────────────────────────


@pytest.mark.parametrize("fam_name", ["nb", "zinb"])
def test_generate_posterior_samples_dxk_fill(fam_name):
    d, k = 2, 3
    inf = _make_inference(fam_name, n_posterior_samples=d, n_head_samples=k)
    handler = _mock_handler(_FEATURES)
    mag, prob = inf.generate_posterior_samples(handler, origin=1)
    # cube [T, H, W, n_reg, S] with S = D*K
    assert mag.shape == (1, 8, 8, 3, d * k)
    assert prob.shape == (1, 8, 8, 3, d * k)
    assert np.isfinite(mag).all()


def test_generate_posterior_samples_family_deterministic():
    inf = _make_inference("nb", n_posterior_samples=2, n_head_samples=3)
    handler = _mock_handler(_FEATURES)  # SAME input both runs: determinism = fixed seed + input
    mag1, _ = inf.generate_posterior_samples(handler, origin=1)
    mag2, _ = inf.generate_posterior_samples(handler, origin=1)
    np.testing.assert_array_equal(mag1, mag2)


def test_generate_posterior_samples_legacy_width_unchanged():
    # standard head, K=1 -> S == n_posterior_samples (byte-identical width, Path B)
    inf = _make_inference("standard", n_posterior_samples=4, n_head_samples=1)
    handler = _mock_handler(_FEATURES)
    mag, _ = inf.generate_posterior_samples(handler, origin=1)
    assert mag.shape == (1, 8, 8, 3, 4)
