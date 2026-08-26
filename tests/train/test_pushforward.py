"""Pushforward (#289, Brandstetter et al. 2022) — prove the mechanism, not the shape.

Every test here exists because a plausible-looking implementation could pass without doing the
thing. The specific traps, all real:

* **The gradient cut is already free on our `sample` path.** `torch.poisson` severs it —
  measured as
  exactly 0.0 while the tensor stays graph-connected. An implementation consisting of a `.detach()`
  would be a NO-OP that looks correct, so the tests below check the SECOND UNROLL, which is the
  entire intervention here.
* **The second step must consume the model's OWN field, not ground truth.** Feeding `t0`
  again would
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
from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices  # noqa: E402

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
        "features": FEATS,
        "regression_targets": FEATS,
        "classification_targets": CLS,
        "static_channels": [],
    }
    names = FEATS + CLS
    idx = _SequenceIndices(names, cfg)
    model = (
        get_architecture("HydraBNUNet06_LSTM4")(len(FEATS), 32, 1, 0.0, output_distribution="nb")
        .float()
        .train()
    )
    h = model.init_hTtime(model.base, H, W).float()
    # log1p-space counts, mostly zero like the real grid
    x = (
        (torch.rand(1, T, len(names), H, W) < 0.05).float()
        * torch.rand(1, T, len(names), H, W)
        * 3
    )
    return x, model, h, idx, get_family("nb")


def _run(pf_weight=0.0, detach_state=False, seed=0, tensor=None, **kw):
    x, model, h, idx, fam = _fixture(seed)
    if tensor is not None:
        x = tensor
    out = _process_sequence(
        x,
        model,
        h,
        criterion_reg=FamilyLoss(fam),
        criterion_class=nn.BCEWithLogitsLoss(),
        multitaskloss_instance=_MTL(),
        idx=idx,
        device=torch.device("cpu"),
        family=fam,
        ss_feedback="sample",
        forecast_composition="soft_gate",
        pushforward_weight=pf_weight,
        pushforward_detach_state=detach_state,
        **kw,
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
    that left that line in place and fed `t0_input` to the model instead **passed all seven
    tests** — the string was present, the property was gone. That is the same grep-the-source
    failure the project keeps registering, so this recomputes the term from first principles.

    Uses `ss_feedback='mean'`, which is analytic and needs no RNG, so the reference is exact. The
    plumbing under test — WHICH tensor reaches the second forward — is identical on both paths.

    The field here is deliberately DENSER and larger than the real grid. This test is about
    plumbing, not about a realistic regime, and on a near-empty field every candidate input
    produces nearly the same loss, leaving too little margin to tell a correct implementation from
    a wrong one. Measured on this fixture: feeding the model's own field differs from feeding `t0`
    by 2.8e-2 and from feeding ground truth at t+1 by 1.3e-2, both ~100x the 1e-4 match tolerance.
    """
    from views_hydranet.train.training_engine import (
        _attach_static_channels,
        _batchnorm_eval,
        _family_feedback_log1p,
    )

    x, model, h, idx, fam = _fixture(seed=21)
    torch.manual_seed(21)
    short = (torch.rand(1, 3, 6, H, W) < 0.25).float() * torch.rand(1, 3, 6, H, W) * 8
    crit = FamilyLoss(fam)

    def run(w):
        m, hh = _fixture(seed=21)[1], _fixture(seed=21)[2]
        return _process_sequence(
            short,
            m,
            hh,
            criterion_reg=crit,
            criterion_class=nn.BCEWithLogitsLoss(),
            multitaskloss_instance=_MTL(),
            idx=idx,
            device=torch.device("cpu"),
            family=fam,
            ss_feedback="mean",
            forecast_composition="soft_gate",
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
        # mirrors the engine: the extra forward runs with BN frozen so it cannot write running
        # statistics (see the C-184 note in _process_sequence)
        with _batchnorm_eval(m):
            o = m(inp, state)
        y2 = short[:, 2, idx.reg]
        n = crit.n_params
        return sum(
            crit(o.reg[:, j * n : (j + 1) * n].permute(0, 2, 3, 1), y2[:, j])
            for j in range(len(idx.reg_names))
        ).item()

    t1 = short[:, 1]
    from_own_field = pf_from(_attach_static_channels(fed, t1, idx), out0.h_next)
    # the two implementations this must rule out
    fed_t0_again = pf_from(t0_input, out0.h_next)
    fed_truth_at_t1 = pf_from(_attach_static_channels(t1[:, idx.feat], t1, idx), out0.h_next)

    assert observed == pytest.approx(from_own_field, rel=1e-4), (
        f"the pushforward term ({observed:.6f}) does not match a second step fed the model's own "
        f"field ({from_own_field:.6f})"
    )
    for label, wrong in (("t0 again", fed_t0_again), ("ground truth at t+1", fed_truth_at_t1)):
        margin = abs(from_own_field - wrong) / abs(from_own_field)
        assert margin > 1e-2, (
            f"feeding {label} gives nearly the same loss as the model's own field "
            f"(relative gap {margin:.2e}), so the assertion above could not tell them apart — "
            "the test would be vacuous on this fixture"
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
    """The recurrent fork: detaching `h` must alter gradients but not the forward value.

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


def _bn_state(model):
    """Snapshot every BatchNorm running buffer, so a test can prove none of them moved."""
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            out[name] = (
                m.running_mean.clone(),
                m.running_var.clone(),
                m.num_batches_tracked.clone(),
            )
    return out


def test_the_extra_forward_does_not_touch_batchnorm_running_statistics():
    """The pushforward must not write model STATE — only a loss term.

    Found by review, and it was real: the extra ``model(...)`` call runs while the model is in
    ``train()`` mode, so it updated ``running_mean``/``running_var``/``num_batches_tracked`` on all
    15 BN layers. Measured before the fix on a T=6 window: ``num_batches_tracked`` 5 -> 9, and
    ``bn_enc_conv0.running_mean`` moved by 0.054. Roughly half the saved BN statistics would have
    come from self-fed, off-distribution inputs.

    That is not a cosmetic leak. BN buffers go into the artifact and are used at eval, and the
    C-184 recalibration (``bn_recalibrate: True`` by default) recomputes them with
    ``momentum=None`` — a cumulative average, so the pushforward forwards would carry EQUAL weight.
    An arm with ``pushforward_weight > 0`` would then differ from its ``0.0`` control at the BN
    layer for reasons having nothing to do with the auxiliary loss: the A/B confounded, silently.

    ``torch.no_grad()`` is not a defence — it stops gradients, not buffer updates.
    """
    counts = {}
    for weight in (0.0, 0.5):
        x, model, h, idx, fam = _fixture(seed=11)
        before = _bn_state(model)
        _process_sequence(
            x,
            model,
            h,
            criterion_reg=FamilyLoss(fam),
            criterion_class=nn.BCEWithLogitsLoss(),
            multitaskloss_instance=_MTL(),
            idx=idx,
            device=torch.device("cpu"),
            family=fam,
            ss_feedback="mean",
            forecast_composition="soft_gate",
            pushforward_weight=weight,
        )
        after = _bn_state(model)
        assert before, "the fixture model has no BatchNorm layers — this test cannot bite"
        counts[weight] = {k: int(v[2]) for k, v in after.items()}

    assert counts[0.0] == counts[0.5], (
        "the pushforward changed how many batches BatchNorm has seen "
        f"({counts[0.0]} vs {counts[0.5]}). Its extra forward is updating BN running statistics, "
        "which are model state saved into the artifact and recomputed by the C-184 recalibration "
        "— an arm would differ from its control at the BN layer for reasons unrelated to the loss."
    )


def test_the_pushforward_is_skipped_when_gradients_are_off():
    """Under ``no_grad`` the term is computed and thrown away — that is pure cost.

    ``_recalibrate_bn`` drives the same ``train()`` code path under ``no_grad`` to recompute BN
    statistics. Nothing there can consume a loss, so the extra forward buys nothing and costs a
    full second pass per step.
    """
    x, model, h, idx, fam = _fixture(seed=13)
    kwargs = dict(
        criterion_reg=FamilyLoss(fam),
        criterion_class=nn.BCEWithLogitsLoss(),
        multitaskloss_instance=_MTL(),
        idx=idx,
        device=torch.device("cpu"),
        family=fam,
        ss_feedback="mean",
        forecast_composition="soft_gate",
    )
    with torch.no_grad():
        off = _process_sequence(x, model, h, pushforward_weight=0.0, **kwargs)["total"]
    x2, model2, h2, _, _ = _fixture(seed=13)
    with torch.no_grad():
        on = _process_sequence(x2, model2, h2, pushforward_weight=1.0, **kwargs)["total"]

    assert torch.equal(off, on), (
        f"under no_grad, pushforward_weight=1.0 changed the loss ({on.item()} vs {off.item()}). "
        "The term is still being computed on a path that cannot use it — pure cost during BN "
        "recalibration."
    )
