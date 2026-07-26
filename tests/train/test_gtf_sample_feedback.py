"""EXP-4 / GTF (bloom dossier): scheduled sampling feeds back a composition-aware SAMPLE.

Scheduled sampling (ADR-056) mixes ground-truth vs a fed-back copy per cell with probability e.
Before EXP-4 the fed-back copy for a FAMILY head was the raw n_params channels (shape-mismatched to
the n_reg dynamic inputs — untested). EXP-4 collapses it to the per-target log1p **mean** ('mean')
or a composition-aware **draw** ('sample', so train-exposure == deploy-exposure). e=0 stays
byte-identical (branch never runs). Legacy point heads keep raw-pred feedback.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.distributions import resolve_family  # noqa: E402
from views_hydranet.train.training_engine import _family_feedback_log1p  # noqa: E402


def _params(fam, b=2, h=6, w=6, n_reg=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    npar = fam.n_params
    cols = []
    for _ in range(n_reg):
        raw = torch.randn(b, h, w, npar, generator=g)
        cols.append(fam.activate(raw).permute(0, 3, 1, 2))  # [b,npar,h,w]
    return torch.cat(cols, dim=1)  # [b, n_reg*npar, h, w]


def test_feedback_mean_collapses_to_n_reg_channels():
    """Family emits n_reg*n_params channels; 'mean' feedback must be n_reg (matches dyn inputs)."""
    fam = resolve_family("nb")
    reg = _params(fam)  # [2, 3*2, 6, 6]
    gate = torch.rand(2, 3, 6, 6)
    fb = _family_feedback_log1p(reg, fam, "mean", gate, "self_zeroed", None)
    assert fb.shape == (2, 3, 6, 6)  # n_reg channels, not n_reg*n_params
    assert (fb >= 0).all()  # log1p space


def test_feedback_sample_shape_and_log1p_and_differs_from_mean():
    fam = resolve_family("nb")
    reg = _params(fam)
    gate = torch.rand(2, 3, 6, 6)
    torch.manual_seed(0)
    fb_s = _family_feedback_log1p(reg, fam, "sample", gate, "self_zeroed", None)
    fb_m = _family_feedback_log1p(reg, fam, "mean", gate, "self_zeroed", None)
    assert fb_s.shape == (2, 3, 6, 6) and (fb_s >= 0).all()
    assert not torch.equal(fb_s, fb_m)  # a draw is not the mean


def test_feedback_sample_composes_with_gate_soft():
    """soft_gate zeros some draws (gate<Bernoulli) → differs from the self_zeroed draw."""
    fam = resolve_family("nb")
    reg = _params(fam)
    gate = torch.full((2, 3, 6, 6), 0.5)
    torch.manual_seed(0)
    fb_self = _family_feedback_log1p(reg, fam, "sample", gate, "self_zeroed", None)
    torch.manual_seed(0)
    fb_soft = _family_feedback_log1p(reg, fam, "sample", gate, "soft_gate", None)
    assert fb_soft.shape == fb_self.shape
    assert not torch.equal(fb_soft, fb_self)  # the gate masked some cells to zero


def test_feedback_zinb_self_zeroes_natively():
    fam = resolve_family("zinb")
    reg = _params(fam)
    gate = torch.rand(2, 3, 6, 6)
    fb = _family_feedback_log1p(reg, fam, "sample", gate, "self_zeroed", None)
    assert fb.shape == (2, 3, 6, 6) and (fb >= 0).all()
