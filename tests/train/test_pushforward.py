"""Pushforward (#289, Brandstetter et al. 2022) — prove the mechanism, not the shape.

Every test here exists because a plausible-looking implementation could pass without doing the
thing. The specific traps, all real:

* **The gradient cut is already free on our `sample` path.** `torch.poisson` severs it — measured as
  exactly 0.0 while the tensor stays graph-connected. An implementation consisting of a `.detach()`
  would be a NO-OP that looks correct, so the tests below check the SECOND UNROLL, which is the
  entire intervention here.
* **The second step must consume the model's OWN field, not ground truth.** Feeding `t0` again would
  train an ordinary two-step teacher-forced model and still change the loss.
* **The loss must be scored at t+2.** Scoring it at t+1 would duplicate the main term and still
  "work".
* **Default-off must be byte-identical**, or every existing result is on a different objective.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from views_hydranet.architectures.registry import get_architecture  # noqa: E402
from views_hydranet.distributions import get_family  # noqa: E402
from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402
from views_hydranet.train.training_engine import _SequenceIndices, _process_sequence  # noqa: E402

H = W = 8
T = 6
FEATS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
CLS = ["by_sb_best", "by_ns_best", "by_os_best"]


class _MTL(nn.Module):
    """Stand-in for the multitask loss: a plain sum, so the test reads the real terms."""

    def forward(self, losses):
        return losses.sum()


def _fixture(seed=0):
    torch.manual_seed(seed)
    cfg = {
        "features": FEATS, "regression_targets": FEATS, "classification_targets": CLS,
        "static_channels": [],
    }
    names = FEATS + CLS
    idx = _SequenceIndices(names, cfg)
    model = get_architecture("HydraBNUNet06_LSTM4")(
        len(FEATS), 32, 1, 0.0, output_distribution="nb"
    ).float().train()
    h = model.init_hTtime(model.base, H, W).float()
    # log1p-space counts, mostly zero like the real grid
    x = (torch.rand(1, T, len(names), H, W) < 0.05).float() * torch.rand(1, T, len(names), H, W) * 3
    return x, model, h, idx, get_family("nb")


def _run(pf_weight=0.0, detach_state=False, seed=0, tensor=None, **kw):
    x, model, h, idx, fam = _fixture(seed)
    if tensor is not None:
        x = tensor
    out = _process_sequence(
        x, model, h,
        criterion_reg=FamilyLoss(fam), criterion_class=nn.BCEWithLogitsLoss(),
        multitaskloss_instance=_MTL(), idx=idx, device=torch.device("cpu"),
        family=fam, ss_feedback="sample", forecast_composition="soft_gate",
        pushforward_weight=pf_weight, pushforward_detach_state=detach_state, **kw,
    )
    return out, model, x


def test_default_off_is_byte_identical():
    """weight=0 must not merely be 'close' — the term must never be computed."""
    a, _, _ = _run(pf_weight=0.0, seed=3)
    b, _, _ = _run(pf_weight=0.0, seed=3)
    assert torch.equal(a["total"], b["total"]), "the run is not deterministic"

    # and a weighted run must actually DIFFER, or the flag does nothing
    c, _, _ = _run(pf_weight=1.0, seed=3)
    assert not torch.equal(a["total"], c["total"]), (
        "pushforward_weight=1.0 produced the same loss as 0.0 — the term is inert"
    )


def test_the_term_scales_with_its_weight():
    """A weight that does not move the loss monotonically is not being applied as a weight."""
    base, _, _ = _run(pf_weight=0.0, seed=5)
    one, _, _ = _run(pf_weight=1.0, seed=5)
    two, _, _ = _run(pf_weight=2.0, seed=5)
    d1 = (one["total"] - base["total"]).item()
    d2 = (two["total"] - base["total"]).item()
    assert d1 > 0, "the pushforward term is not positive"
    assert d2 == pytest.approx(2 * d1, rel=1e-4), (
        f"the term does not scale linearly with the weight: {d1} then {d2}"
    )


def test_the_loss_is_scored_at_t_plus_2_not_t_plus_1():
    """THE discriminating test — and the first version of it did NOT discriminate.

    Perturbing a middle frame cannot separate the two: every frame is a t+1 target for one step and
    a t+2 target for another, so the term responds either way. A mutation scoring at t+1 passed.

    With a 3-frame sequence there are two steps, and pushforward fires only at step 0 (it needs
    `i + 2 < seq_len`), scoring frame 2. **Frame 1 is then a t+1 target and never a t+2 target.**
    So perturbing frame 1 alone must leave the pushforward term EXACTLY unchanged — while an
    implementation scoring at t+1 would move.
    """
    x, model, h, idx, fam = _fixture(seed=7)
    short = x[:, :3].clone()

    def pf_term(tensor):
        with_pf, _, _ = _run(pf_weight=1.0, seed=7, tensor=tensor)
        without, _, _ = _run(pf_weight=0.0, seed=7, tensor=tensor)
        return (with_pf["total"] - without["total"]).item()

    # The term is isolated by subtracting two large losses, so float32 cancellation sets the
    # floor: an unchanged term still wobbles ~1e-6 on values ~2. The discriminator is therefore
    # "wobbles" vs "moves substantially", not exact equality.
    base = pf_term(short)

    only_t1 = short.clone()
    only_t1[:, 1, :3] += 3.0  # frame 1: a t+1 target, NEVER a t+2 target
    drift = abs(pf_term(only_t1) - base) / abs(base)
    assert drift < 1e-4, (
        f"the pushforward term moved {drift:.2e} (relative) when only a t+1 target changed — "
        "it is scored at t+1, not t+2"
    )

    only_t2 = short.clone()
    only_t2[:, 2, :3] += 3.0  # frame 2: the t+2 target for step 0
    moved = abs(pf_term(only_t2) - base) / abs(base)
    assert moved > 1e-2, (
        f"the pushforward term moved only {moved:.2e} (relative) when its own t+2 target changed"
    )


def test_the_second_step_consumes_the_models_own_field_not_ground_truth():
    """Recompute the pushforward term independently and require an exact match.

    The previous version of this test grepped the source for `_family_feedback_log1p(`. A mutation
    that left that line in place and fed `t0_input` to the model instead **passed all seven tests** —
    the string was present, the property was gone. That is the same grep-the-source failure the
    project keeps registering, so this recomputes the term from first principles instead.

    Uses `ss_feedback='mean'`, which is analytic and needs no RNG, so the reference is exact. The
    plumbing under test — WHICH tensor reaches the second forward — is identical on both paths.
    """
    from views_hydranet.train.training_engine import (
        _attach_static_channels,
        _family_feedback_log1p,
    )

    x, model, h, idx, fam = _fixture(seed=21)
    short = x[:, :3].clone()
    crit = FamilyLoss(fam)

    def run(w):
        m, hh = _fixture(seed=21)[1], _fixture(seed=21)[2]
        return _process_sequence(
            short, m, hh, criterion_reg=crit, criterion_class=nn.BCEWithLogitsLoss(),
            multitaskloss_instance=_MTL(), idx=idx, device=torch.device("cpu"),
            family=fam, ss_feedback="mean", forecast_composition="soft_gate",
            pushforward_weight=w,
        )["total"]

    observed = (run(1.0) - run(0.0)).item()

    # --- independent reference: step 0 only, since `i + 2 < 3` fires just once ---
    m, hh = _fixture(seed=21)[1], _fixture(seed=21)[2]
    t0 = short[:, 0]
    t0_input = _attach_static_channels(t0[:, idx.feat], t0, idx)
    out0 = m(t0_input, hh)
    fed = _family_feedback_log1p(
        out0.reg, fam, "mean", torch.sigmoid(out0.cls), "soft_gate", None
    ).detach()

    def pf_from(inp, state):
        o = m(inp, state)
        y2 = short[:, 2, idx.reg]
        n = crit.n_params
        return sum(
            crit(o.reg[:, j * n : (j + 1) * n].permute(0, 2, 3, 1), y2[:, j])
            for j in range(len(idx.reg_names))
        ).item()

    from_own_field = pf_from(_attach_static_channels(fed, t0, idx), out0.h_next)
    from_ground_truth = pf_from(t0_input, out0.h_next)

    assert observed == pytest.approx(from_own_field, rel=1e-4), (
        f"the pushforward term ({observed:.6f}) does not match a second step fed the model's own "
        f"field ({from_own_field:.6f})"
    )
    # and the two must be far enough apart that the check above could ever fail
    assert abs(from_own_field - from_ground_truth) / abs(from_own_field) > 1e-2, (
        "feeding ground truth and feeding the model's own field give nearly the same loss here, "
        "so this fixture cannot discriminate — the test would be vacuous"
    )


def test_gradient_into_the_fed_field_is_cut():
    """The fed field must carry no gradient — measured, not asserted from the source.

    On the `sample` path this is already true before any `.detach()` (torch.poisson severs it), so
    this pins the property rather than the line of code that is supposed to produce it.
    """
    from views_hydranet.train.training_engine import _family_feedback_log1p

    fam = get_family("nb")
    reg = (torch.rand(1, 6, H, W) + 0.5).requires_grad_(True)
    gate = torch.rand(1, 3, H, W)
    fed = _family_feedback_log1p(reg, fam, "sample", gate, "soft_gate", None)
    fed.sum().backward()
    assert reg.grad is None or float(reg.grad.abs().sum()) == 0.0, (
        "gradient reaches the params through the sampled feedback — the cut is not in effect"
    )


def test_detach_state_changes_the_gradient_but_not_the_loss():
    """The recurrent fork: detaching `h` must alter gradients while leaving the forward value alone.

    If the two settings produced identical gradients the flag would be decorative; if they produced
    different losses it would be doing something other than cutting a gradient path.
    """
    lo, model_a, _ = _run(pf_weight=1.0, detach_state=False, seed=11)
    hi, model_b, _ = _run(pf_weight=1.0, detach_state=True, seed=11)
    assert lo["total"].item() == pytest.approx(hi["total"].item(), rel=1e-6), (
        "detaching the state changed the forward loss; it must only change gradient flow"
    )

    lo["total"].backward()
    hi["total"].backward()
    ga = torch.cat([p.grad.flatten() for p in model_a.parameters() if p.grad is not None])
    gb = torch.cat([p.grad.flatten() for p in model_b.parameters() if p.grad is not None])
    assert not torch.allclose(ga, gb, atol=1e-8), (
        "detaching the recurrent state left gradients unchanged — the flag is decorative"
    )


def test_the_last_step_is_skipped_rather_than_indexing_past_the_end():
    """`i + 2 < seq_len` must hold; an off-by-one here would crash mid-training, hours in."""
    out, _, _ = _run(pf_weight=1.0, seed=13)
    assert torch.isfinite(out["total"]).all()
