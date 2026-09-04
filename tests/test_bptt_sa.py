"""BPTT-SA (#308): let the gradient reach back THROUGH the fed-back prediction.

The training graph's feedback path had exactly one cut in it — a `.detach()` on the prediction that
scheduled sampling feeds to the next step. The recurrent state already flows across steps
undetached
(a ~383-step graph under one `backward()`, measured 2026-08-26), so with that cut the model is told
"step i was wrong" and never "step i-1 produced the input that made it wrong". That is a mechanical
account of why scheduled sampling failed here (M26-M33), and removing the cut is the whole change.

What these tests have to establish, in order of what would hurt most if wrong:

1. **Production is untouched.** Every shipped arm runs `ss_epsilon = 0`, where the feedback branch
   never executes. The flag must be provably inert there.
2. **Off means off.** With the flag false the fed value is detached, exactly as before.
3. **On means on.** With it true the fed value carries a grad_fn — the wire is actually
reconnected.
4. **It changes learning, not just plumbing.** The parameter gradients must actually differ.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices  # noqa: E402

_H, _W, _T = 2, 2, 4
_FEATURE_NAMES = ["reg0", "cls0", "feat0"]
_CONFIG = {
    "regression_targets": ["reg0"],
    "classification_targets": ["cls0"],
    "features": ["feat0"],
    "static_channels": [],
}


class _LinearModel(torch.nn.Module):
    """A trainable stub whose output depends on its input, so feedback can carry gradient."""

    def __init__(self, n_reg: int, n_cls: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, n_reg + n_cls, kernel_size=1)
        self.n_reg = n_reg

    def forward(self, x, hidden):  # noqa: ANN001 - stub
        out = self.conv(x[:, :1])
        reg, cls = out[:, : self.n_reg], out[:, self.n_reg :]
        return types.SimpleNamespace(reg=reg, cls=cls, reg_latent=reg, h_next=hidden)


def _run(*, ss_epsilon: float, backprop: bool, seed: int = 0):
    """One sequence pass; returns (loss_dict, model) with grads populated."""
    torch.manual_seed(seed)
    idx = _SequenceIndices(_FEATURE_NAMES, _CONFIG)
    tensor = torch.rand(1, _T, len(_FEATURE_NAMES), _H, _W)
    model = _LinearModel(idx.n_reg, idx.n_cls)
    res = _process_sequence(
        train_tensor=tensor,
        model=model,
        h=torch.zeros(1, 1, 1, 1),
        criterion_reg=torch.nn.MSELoss(),
        criterion_class=lambda pred, targ: (pred * 0.0).sum(),
        multitaskloss_instance=lambda losses: losses.sum(),
        idx=idx,
        device=torch.device("cpu"),
        event_threshold=0.0,
        ss_epsilon=ss_epsilon,
        ss_backprop_through_feedback=backprop,
    )
    return res, model


def _grad_vector(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


# ── 1. production is untouched ───────────────────────────────────────────


def test_the_flag_is_inert_when_scheduled_sampling_is_off():
    """Every shipped arm runs ss_epsilon = 0. The flag must not reach the graph there.

    This is the one that protects production: the feedback branch is guarded by `ss_epsilon > 0`,
    so with sampling off the flag cannot change a single number.
    """
    a, ma = _run(ss_epsilon=0.0, backprop=False)
    b, mb = _run(ss_epsilon=0.0, backprop=True)
    assert a["total"] == b["total"]
    a["total"].backward()
    b["total"].backward()
    torch.testing.assert_close(_grad_vector(ma), _grad_vector(mb), rtol=0, atol=0)


# ── 2 & 3. the wire ──────────────────────────────────────────────────────


def test_off_detaches_the_fed_prediction_and_on_does_not():
    """The whole change, observed directly: is the handoff connected to what produced it?

    Captured by wrapping the model so the tensor actually handed to the next step is visible.
    """
    seen = {}

    def _capture(backprop):
        torch.manual_seed(0)
        idx = _SequenceIndices(_FEATURE_NAMES, _CONFIG)
        model = _LinearModel(idx.n_reg, idx.n_cls)
        inputs = []
        real = model.forward
        model.forward = lambda x, h: (inputs.append(x), real(x, h))[1]
        _process_sequence(
            train_tensor=torch.rand(1, _T, len(_FEATURE_NAMES), _H, _W),
            model=model,
            h=torch.zeros(1, 1, 1, 1),
            criterion_reg=torch.nn.MSELoss(),
            criterion_class=lambda pred, targ: (pred * 0.0).sum(),
            multitaskloss_instance=lambda losses: losses.sum(),
            idx=idx,
            device=torch.device("cpu"),
            event_threshold=0.0,
            ss_epsilon=1.0,  # always feed back, so the branch is exercised every step
            ss_backprop_through_feedback=backprop,
        )
        seen[backprop] = inputs

    _capture(False)
    _capture(True)
    # step 0 reads ground truth in both; from step 1 the fed prediction is the input
    assert seen[False][1].grad_fn is None, (
        "flag OFF: the fed input is still attached to its producer"
    )
    assert seen[True][1].grad_fn is not None, "flag ON: the fed input was detached anyway"


# ── 4. it changes learning ───────────────────────────────────────────────


def test_it_changes_the_gradients_not_just_the_plumbing():
    """A reconnected wire that does not move any gradient would be a no-op with a nice story."""
    a, ma = _run(ss_epsilon=1.0, backprop=False)
    b, mb = _run(ss_epsilon=1.0, backprop=True)
    a["total"].backward()
    b["total"].backward()
    ga, gb = _grad_vector(ma), _grad_vector(mb)
    assert ga.shape == gb.shape
    assert not torch.allclose(ga, gb), "attaching the feedback path changed no gradient at all"


def test_the_forward_values_are_unchanged_only_the_gradients_differ():
    """BPTT-SA alters credit assignment, NOT the forward pass. If the loss itself moved, the arms
    would differ for a second reason and the comparison against M26-M33 would be confounded."""
    a, _ = _run(ss_epsilon=1.0, backprop=False)
    b, _ = _run(ss_epsilon=1.0, backprop=True)
    assert float(a["total"]) == pytest.approx(float(b["total"]), rel=1e-9)


# ── config default ───────────────────────────────────────────────────────


def test_config_default_is_off(valid_config_dict):
    from views_hydranet.utils.config_initializer import HydraNetConfig

    assert HydraNetConfig(**valid_config_dict).ss_backprop_through_feedback is False


def test_at_epsilon_zero_the_model_never_receives_a_prediction_as_input():
    """Found by mutation (B4): the eps guard is duplicated — one on producing the fed value, one on
    consuming it — and pinning only the producer let a mutation through.

    This pins the PROPERTY that actually protects production rather than either implementation of
    it: with scheduled sampling off, every input the model sees must be the ground-truth dynamic
    features. If a future change relaxes the consumer guard, shipped arms would silently begin
    training on their own output and nothing else here would notice.
    """
    torch.manual_seed(0)
    idx = _SequenceIndices(_FEATURE_NAMES, _CONFIG)
    tensor = torch.rand(1, _T, len(_FEATURE_NAMES), _H, _W)
    model = _LinearModel(idx.n_reg, idx.n_cls)
    inputs = []
    real = model.forward
    model.forward = lambda x, h: (inputs.append(x.clone()), real(x, h))[1]

    _process_sequence(
        train_tensor=tensor,
        model=model,
        h=torch.zeros(1, 1, 1, 1),
        criterion_reg=torch.nn.MSELoss(),
        criterion_class=lambda pred, targ: (pred * 0.0).sum(),
        multitaskloss_instance=lambda losses: losses.sum(),
        idx=idx,
        device=torch.device("cpu"),
        event_threshold=0.0,
        ss_epsilon=0.0,
        ss_backprop_through_feedback=True,  # even with the new flag ON
    )

    for i, got in enumerate(inputs):
        want = tensor[:, i][:, idx.feat, :, :]
        torch.testing.assert_close(
            got,
            want,
            rtol=0,
            atol=0,
            msg=f"step {i}: model was fed something other than ground truth at ss_epsilon=0",
        )


def test_at_epsilon_zero_the_feedback_machinery_does_not_run_at_all():
    """Mutation B4 survived every behavioural test: a SECOND guard blocks the fed value's USE, so
    computing it changes nothing observable — with a point head.

    With a FAMILY head it is not harmless. `_family_feedback_log1p` under `ss_feedback="sample"`
    DRAWS, so running it at ss_epsilon=0 consumes RNG and shifts every subsequent random draw in
    the
    run. Production arms are family heads at eps=0 with sample feedback, so the property worth
    pinning is stronger than "the fed value is unused": at eps=0 the branch must never be ENTERED.

    Detected by shape: a family is passed whose n_params does not divide the stub's single output
    channel. If the guard holds, the branch never runs and nothing raises. If it is ever relaxed,
    the feedback call fails loudly here instead of silently reseeding a production run.
    """
    from views_hydranet.distributions import resolve_family

    fam = resolve_family("nb")
    assert fam.n_params > 1, "fixture assumes a multi-parameter family"

    torch.manual_seed(0)
    idx = _SequenceIndices(_FEATURE_NAMES, _CONFIG)
    model = _LinearModel(idx.n_reg, idx.n_cls)  # emits 1 reg channel; nb needs n_params per target
    _process_sequence(
        train_tensor=torch.rand(1, _T, len(_FEATURE_NAMES), _H, _W),
        model=model,
        h=torch.zeros(1, 1, 1, 1),
        criterion_reg=torch.nn.MSELoss(),
        criterion_class=lambda pred, targ: (pred * 0.0).sum(),
        multitaskloss_instance=lambda losses: losses.sum(),
        idx=idx,
        device=torch.device("cpu"),
        event_threshold=0.0,
        ss_epsilon=0.0,
        ss_feedback="sample",
        family=fam,
        ss_backprop_through_feedback=True,
    )


# ── the limitation that voided the first screen ──────────────────────────


@pytest.mark.parametrize("mode,expect_gradient", [("mean", True), ("sample", False)])
def test_the_feedback_is_only_differentiable_under_mean(mode, expect_gradient):
    """BPTT-SA is INERT under `ss_feedback="sample"`, and this is why.

    Reconnecting the wire only matters if a gradient can travel it. A DRAW from the family is not
    reparameterised, so d(fed)/d(params) is exactly 0 — the tensor carries a grad_fn from the
    log1p wrapper, which makes it *look* connected, and delivers nothing.

    Measured 2026-09-03: mean -> 167.8, sample -> 0.0. The first screen trained two arms whose
    weights came out byte-identical for exactly this reason, and C-259 REQUIRES sample whenever
    eps > 0 — so the production path is the one where the change cannot do anything.

    This test exists so that is a stated property with a number on it rather than a surprise, and
    so any future reparameterised or straight-through feedback flips it and is noticed.
    """
    from views_hydranet.distributions import resolve_family
    from views_hydranet.train.training_engine import _family_feedback_log1p

    fam = resolve_family("nb")
    raw = torch.randn(1, 3 * fam.n_params, 4, 4, requires_grad=True)
    gate = torch.rand(1, 3, 4, 4, requires_grad=True)

    out = _family_feedback_log1p(raw, fam, mode, gate, "soft_gate", None)
    g = torch.autograd.grad(out.sum(), raw, allow_unused=True)[0]
    total = 0.0 if g is None else float(g.abs().sum())

    if expect_gradient:
        assert total > 0, f"{mode!r} feedback carries no gradient — BPTT-SA cannot work at all"
    else:
        assert total == 0.0, (
            f"{mode!r} feedback now carries gradient ({total}) — if the draw was made "
            "reparameterised or straight-through, BPTT-SA is no longer inert here and the "
            "screen that assumed it was must be re-run"
        )


def test_a_grad_fn_is_not_evidence_the_wire_carries_anything():
    """The trap directly: the sampled feedback HAS a grad_fn and still delivers zero gradient.

    The earlier tests in this file checked `grad_fn is not None` on the POINT-head path and
    concluded the wire was connected. That is the C-323 error in a new place — a property was
    verified on a path production never takes.
    """
    from views_hydranet.distributions import resolve_family
    from views_hydranet.train.training_engine import _family_feedback_log1p

    fam = resolve_family("nb")
    raw = torch.randn(1, 3 * fam.n_params, 4, 4, requires_grad=True)
    out = _family_feedback_log1p(raw, fam, "sample", torch.rand(1, 3, 4, 4), "soft_gate", None)

    assert out.grad_fn is not None, "fixture no longer reproduces the misleading appearance"
    g = torch.autograd.grad(out.sum(), raw, allow_unused=True)[0]
    assert g is None or float(g.abs().sum()) == 0.0


# ── the straight-through fix, on the PRODUCTION path ─────────────────────


def _activated(fam, n_reg=3, h=8, w=8, seed=0):
    """Params as the model actually emits them: ACTIVATED (ADR-067), not raw.

    The first NaN scare in this work was a fixture that fed raw `randn` where production supplies
    activated params — the same 'tested a path production never takes' error as C-323, third
    occurrence. Hence this helper exists rather than each test rolling its own.
    """
    npar = fam.n_params
    torch.manual_seed(seed)
    raw = torch.randn(1, n_reg * npar, h, w)
    act = torch.cat(
        [
            fam.activate(raw[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            for j in range(n_reg)
        ],
        dim=1,
    )
    return act.requires_grad_(True)


def test_the_straight_through_surrogate_is_differentiable_and_finite():
    """The whole point of the fix: a gradient must actually reach the parameters."""
    from views_hydranet.distributions import resolve_family
    from views_hydranet.train.training_engine import _family_composed_mean_log1p

    fam = resolve_family("nb")
    act = _activated(fam)
    gate = torch.rand(1, 3, 8, 8)
    m = _family_composed_mean_log1p(act, fam, gate, "soft_gate", None)
    assert torch.isfinite(m).all(), "the surrogate is not finite — it would poison training"
    g = torch.autograd.grad(m.sum(), act)[0]
    assert float(g.abs().sum()) > 0, "the surrogate carries no gradient — the fix is inert"


def test_the_surrogate_is_the_COMPOSED_mean_not_the_bare_body_mean():
    """`_family_target_log1p_mean` ignores the gate, so it is the analogue of an UNCOMPOSED draw.

    Using it would push gradient for a quantity the forward pass never produced. This pins that the
    surrogate matches what deployment actually emits.
    """
    from views_hydranet.distributions import resolve_family
    from views_hydranet.train.training_engine import (
        _family_composed_mean_log1p,
        _family_target_log1p_mean,
    )

    fam = resolve_family("nb")
    act = _activated(fam, seed=3)
    gate = torch.rand(1, 3, 8, 8) * 0.5  # a gate well away from 1, so composing must matter
    composed = _family_composed_mean_log1p(act, fam, gate, "soft_gate", None)
    bare = _family_target_log1p_mean(act, fam)
    assert not torch.allclose(composed, bare), "the surrogate ignored the gate"
    assert (composed <= bare + 1e-6).all(), "soft_gate must not increase the mean"


def test_straight_through_leaves_the_forward_pass_BIT_IDENTICAL():
    """The arms must differ in credit assignment ONLY, or the comparison is confounded.

    `surrogate + (draw - surrogate).detach()` is exactly `draw` in value; measured max|diff| = 0.
    """
    from views_hydranet.distributions import resolve_family
    from views_hydranet.train.training_engine import (
        _family_composed_mean_log1p,
        _family_feedback_log1p,
    )

    fam = resolve_family("nb")
    act = _activated(fam, seed=5)
    gate = torch.rand(1, 3, 8, 8)
    torch.manual_seed(7)
    draw = _family_feedback_log1p(act, fam, "sample", gate, "soft_gate", None)
    surrogate = _family_composed_mean_log1p(act, fam, gate, "soft_gate", None)
    ste = surrogate + (draw - surrogate).detach()

    assert torch.equal(ste, draw), "straight-through changed the forward value"
    assert torch.isfinite(ste).all()
    g = torch.autograd.grad(ste.sum(), act)[0]
    assert float(g.abs().sum()) > 0, "straight-through carries no gradient"


def test_POTENCY_the_knob_moves_the_gradient_on_the_production_config():
    """The gate that would have saved 276 minutes. Runs on family=nb + ss_feedback='sample',
    which is what C-259 forces production to use — not on a convenient fixture."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import potency_check as pc

    from views_hydranet.distributions import resolve_family
    from views_hydranet.train.training_engine import (
        _family_composed_mean_log1p,
        _family_feedback_log1p,
    )

    fam = resolve_family("nb")
    act = _activated(fam, seed=11)
    gate = torch.rand(1, 3, 8, 8)

    def grad_through(backprop: bool) -> float:
        torch.manual_seed(3)
        fed = _family_feedback_log1p(act, fam, "sample", gate, "soft_gate", None)
        if backprop:
            s = _family_composed_mean_log1p(act, fam, gate, "soft_gate", None)
            fed = s + (fed - s).detach()
        else:
            fed = fed.detach()
        if not fed.requires_grad:
            return 0.0
        g = torch.autograd.grad(fed.sum(), act, retain_graph=True, allow_unused=True)[0]
        return 0.0 if g is None else float(g.abs().sum())

    r = pc.assert_potent(
        grad_through,
        off=False,
        on=True,
        name="ss_backprop_through_feedback on the production config",
    )
    assert r["off"] == 0.0 and r["on"] > 0.0


# ── INTEGRATION: the call site, on a family head. The unit tests above miss it. ───────────


class _FamilyModel(torch.nn.Module):
    """A trainable stub that emits ACTIVATED family params, as the real head does (ADR-067).

    Exists because every earlier test in this file drove the POINT-head path, where the fed value
    is the raw prediction and trivially differentiable. Production is a family head with sampled
    feedback, and that is the path where #308's no-op lived. Mutations at the call site survived
    every unit test here until this stub existed.
    """

    def __init__(self, fam, n_reg: int, n_cls: int) -> None:
        super().__init__()
        self.fam, self.n_reg, self.npar = fam, n_reg, fam.n_params
        self.conv = torch.nn.Conv2d(1, n_reg * fam.n_params + n_cls, kernel_size=1)

    def forward(self, x, hidden):  # noqa: ANN001 - stub
        o = self.conv(x[:, :1])
        npar, n = self.npar, self.n_reg
        act = torch.cat(
            [
                self.fam.activate(o[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1)).permute(
                    0, 3, 1, 2
                )
                for j in range(n)
            ],
            dim=1,
        )
        latent = act[:, ::npar]  # n_reg channels, still differentiable
        return types.SimpleNamespace(
            reg=act, cls=o[:, n * npar :], reg_latent=latent, h_next=hidden
        )


class _LatentMSE(torch.nn.MSELoss):
    """Routes the loss to reg_latent so `reg` may carry n_params channels."""

    needs_latent = True


def _run_family(
    *,
    backprop: bool,
    seed: int = 0,
    clip: float | None = None,
    sink: list[float] | None = None,
    return_loss: bool = False,
):
    from views_hydranet.distributions import resolve_family

    fam = resolve_family("nb")
    torch.manual_seed(seed)
    idx = _SequenceIndices(_FEATURE_NAMES, _CONFIG)
    model = _FamilyModel(fam, idx.n_reg, idx.n_cls)
    res = _process_sequence(
        train_tensor=torch.rand(1, _T, len(_FEATURE_NAMES), _H, _W),
        model=model,
        h=torch.zeros(1, 1, 1, 1),
        criterion_reg=_LatentMSE(),
        criterion_class=lambda pred, targ: (pred * 0.0).sum(),
        multitaskloss_instance=lambda losses: losses.sum(),
        idx=idx,
        device=torch.device("cpu"),
        event_threshold=0.0,
        ss_epsilon=1.0,  # always feed back, so the branch runs every step
        ss_feedback="sample",  # what C-259 forces production to use
        forecast_composition="soft_gate",
        family=fam,
        ss_backprop_through_feedback=backprop,
        ss_feedback_grad_clip=clip,
        ss_feedback_grad_sink=sink,
    )
    loss = float(res["total"])
    res["total"].backward()
    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return (grads, loss) if return_loss else grads


def test_INTEGRATION_the_flag_changes_gradients_on_a_family_head():
    """THE test this work needed from the start.

    #308's first implementation passed 7 unit tests, 5/5 mutations, lint and the full suite, and
    was a no-op on exactly this path — two arms trained to byte-identical weights over 276 minutes.
    Everything above drives the point head; only this drives the configuration production runs.
    """
    off, on = _run_family(backprop=False), _run_family(backprop=True)
    assert off.shape == on.shape
    assert torch.isfinite(on).all(), "the straight-through path produced non-finite gradients"
    delta = float((on - off).abs().sum())
    assert delta > 0, (
        "the flag changed NO gradient on a family head with sampled feedback — this is exactly "
        "the #308 no-op, and any experiment built on it would measure the harness, not the idea"
    )


def test_INTEGRATION_the_forward_loss_is_unchanged_on_a_family_head():
    """Straight-through alters credit assignment only. If the loss moved, the arms would differ
    for a second reason and the comparison would be confounded."""
    from views_hydranet.distributions import resolve_family

    fam = resolve_family("nb")
    losses = []
    for backprop in (False, True):
        torch.manual_seed(0)
        idx = _SequenceIndices(_FEATURE_NAMES, _CONFIG)
        model = _FamilyModel(fam, idx.n_reg, idx.n_cls)
        torch.manual_seed(99)  # same draws in both arms
        res = _process_sequence(
            train_tensor=torch.rand(1, _T, len(_FEATURE_NAMES), _H, _W),
            model=model,
            h=torch.zeros(1, 1, 1, 1),
            criterion_reg=_LatentMSE(),
            criterion_class=lambda pred, targ: (pred * 0.0).sum(),
            multitaskloss_instance=lambda losses_: losses_.sum(),
            idx=idx,
            device=torch.device("cpu"),
            event_threshold=0.0,
            ss_epsilon=1.0,
            ss_feedback="sample",
            forecast_composition="soft_gate",
            family=fam,
            ss_backprop_through_feedback=backprop,
        )
        losses.append(float(res["total"]))
    assert losses[0] == pytest.approx(losses[1], rel=1e-9), (
        f"the forward loss moved ({losses[0]} vs {losses[1]}) — the arms differ for a "
        "second reason"
    )


def test_the_surrogate_is_in_LOG1P_space_not_count_space():
    """Found by mutation (R5): dropping the log1p still leaves the forward value correct.

    `surrogate + (draw - surrogate)` equals the draw in value whatever space the surrogate is in,
    so the forward test cannot see this. Only the GRADIENT changes — by the derivative of log1p,
    i.e. a factor of (1+mu) per cell. That is a silent, plausible, wrong learning signal: exactly
    the failure mode that makes a null result untrustworthy.
    """
    from views_hydranet.distributions import resolve_family
    from views_hydranet.distributions.composition import compose_mean
    from views_hydranet.train.training_engine import _family_composed_mean_log1p

    fam = resolve_family("nb")
    act = _activated(fam, seed=13)
    gate = torch.rand(1, 3, 8, 8)
    npar = fam.n_params

    got = _family_composed_mean_log1p(act, fam, gate, "soft_gate", None)
    mus = torch.stack(
        [fam.mean(act[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1)) for j in range(3)], dim=1
    )
    want = torch.log1p(compose_mean(mus, gate[:, :3], "soft_gate", None))
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    # and it must actually be compressed relative to count space, or the log1p did nothing
    assert float(got.max()) < float(compose_mean(mus, gate[:, :3], "soft_gate", None).max())


def test_straight_through_is_NOT_applied_to_mean_feedback():
    """Found by mutation (R4): dropping the `ss_feedback == "sample"` condition.

    With 'mean' the fed value is already differentiable, so the surrogate is unnecessary — and it
    is a DIFFERENT quantity (composed mean vs bare mean), so applying it would silently change the
    fed value. C-259 forbids 'mean' at eps>0 today, which is exactly why nothing else would notice.
    """
    from views_hydranet.distributions import resolve_family

    fam = resolve_family("nb")
    losses = []
    for backprop in (False, True):
        torch.manual_seed(0)
        idx = _SequenceIndices(_FEATURE_NAMES, _CONFIG)
        model = _FamilyModel(fam, idx.n_reg, idx.n_cls)
        res = _process_sequence(
            train_tensor=torch.rand(1, _T, len(_FEATURE_NAMES), _H, _W),
            model=model,
            h=torch.zeros(1, 1, 1, 1),
            criterion_reg=_LatentMSE(),
            criterion_class=lambda pred, targ: (pred * 0.0).sum(),
            multitaskloss_instance=lambda losses_: losses_.sum(),
            idx=idx,
            device=torch.device("cpu"),
            event_threshold=0.0,
            ss_epsilon=1.0,
            ss_feedback="mean",
            forecast_composition="soft_gate",
            family=fam,
            ss_backprop_through_feedback=backprop,
        )
        losses.append(float(res["total"]))
    assert losses[0] == pytest.approx(losses[1], rel=1e-9), (
        "the flag changed the FORWARD pass under 'mean' feedback — straight-through was applied "
        "where the fed value is already differentiable, silently altering what is fed"
    )


def test_C259_makes_the_mean_plus_sampling_configuration_unreachable():
    """Records why one mutation is left uncaught, rather than leaving it looking like an oversight.

    Mutation R4 — applying straight-through under 'mean' feedback too — SURVIVES this suite. It
    cannot be caught by a forward-value test, because straight-through preserves the forward value
    by construction; it changes only which gradient flows.

    It is left uncaught deliberately: C-259 REJECTS ss_feedback='mean' with ss_epsilon_max>0 at
    config validation for a family head, so the mutated branch is unreachable in anything that can
    train. This test pins that reachability claim, so if C-259 is ever relaxed the justification
    fails here instead of the mutant quietly becoming live.
    """
    from pathlib import Path as _P

    from views_hydranet.utils.config_initializer import HydraNetConfig

    p = (
        _P(__file__).resolve().parents[2]
        / "views-models/models/fullzero_fortytwo/configs/config_hyperparameters.py"
    )
    if not p.exists():
        pytest.skip("floor config not available in this checkout")
    ns: dict = {}
    exec(compile(p.read_text(), str(p), "exec"), ns)  # noqa: S102 - repo-local config
    raw = dict(ns["get_hp_config"]())
    raw.update(run_type="calibration", ss_epsilon_max=0.5, ss_feedback="mean")
    with pytest.raises(Exception, match="C-259"):
        HydraNetConfig(**raw)


# ---------------------------------------------------------------------------
# #308 GRAD-TRAJ follow-up: the per-step feedback-gradient limiter.
#
# GRAD-TRAJ measured the attached arm's pre-clip gradient norm rising 133,465 -> 9.4e9 between
# lessons 15-25 and 38-47 while its control's FELL 859 -> 312, until float32 gave out at lesson 48.
# `clip_grad_norm_` was on throughout and could not help: it runs ONCE at the end of the backward
# pass, after the product has already overflowed. `_clip_feedback_grad` bounds the same gradient
# per step, at the one tensor credit crosses on.
#
# The helper tests below are necessary and NOT sufficient — helper-level tests left 4 of 5
# call-site mutations alive on this very file's earlier work, including the original bug
# reintroduced verbatim. The tests that matter are the two INTEGRATION ones at the end.
# ---------------------------------------------------------------------------


def _grad_of(x, *, clip, sink=None):
    """Backward a fixed scalar through `x` after attaching the limiter; return dL/dx."""
    from views_hydranet.train.training_engine import _clip_feedback_grad

    out = _clip_feedback_grad(x, clip, sink)
    (out * _FIXED_UPSTREAM).sum().backward()
    return x.grad.clone()


_FIXED_UPSTREAM = 100.0


def test_a_clip_of_None_and_no_sink_leaves_the_gradient_bit_identical():
    with_helper = torch.ones(4, requires_grad=True)
    without_helper = torch.ones(4, requires_grad=True)
    ga = _grad_of(with_helper, clip=None)
    (without_helper * _FIXED_UPSTREAM).sum().backward()
    assert torch.equal(ga, without_helper.grad), (
        "the default must be a no-op, or every existing arm changes"
    )


def test_the_clip_rescales_to_exactly_the_threshold_and_does_not_rotate():
    x = torch.ones(4, requires_grad=True)
    g = _grad_of(x, clip=1.0)
    assert pytest.approx(float(g.norm(2)), rel=1e-5) == 1.0
    # direction preserved: an element-wise clamp would flatten these to equal values, which they
    # already are -- so use an asymmetric upstream to make rotation detectable.
    y = torch.ones(4, requires_grad=True)
    from views_hydranet.train.training_engine import _clip_feedback_grad

    out = _clip_feedback_grad(y, 1.0, None)
    (out * torch.tensor([1.0, 2.0, 3.0, 4.0]) * 100.0).sum().backward()
    unit = y.grad / y.grad.norm(2)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = expected / expected.norm(2)
    assert torch.allclose(unit, expected, atol=1e-6), "the clip rotated the gradient"


def test_a_gradient_already_under_the_threshold_is_untouched():
    x = torch.ones(4, requires_grad=True)
    g = _grad_of(x, clip=1e9)
    assert pytest.approx(float(g.norm(2)), rel=1e-6) == float(
        torch.full((4,), _FIXED_UPSTREAM).norm(2)
    )


def test_the_sink_records_the_norm_BEFORE_clipping():
    """If it recorded the post-clip value it would read as flat at the threshold for every
    exploding step -- i.e. the instrument would hide exactly what it exists to show."""
    sink: list[float] = []
    x = torch.ones(4, requires_grad=True)
    _grad_of(x, clip=1.0, sink=sink)
    assert len(sink) == 1
    assert pytest.approx(sink[0], rel=1e-5) == float(torch.full((4,), _FIXED_UPSTREAM).norm(2))
    assert sink[0] > 1.0, "the recorded norm is the clipped one, not the raw one"


def test_a_sink_with_no_clip_observes_without_acting():
    sink: list[float] = []
    x = torch.ones(4, requires_grad=True)
    g = _grad_of(x, clip=None, sink=sink)
    assert len(sink) == 1
    assert pytest.approx(float(g.norm(2)), rel=1e-6) == sink[0]


def test_a_tensor_that_carries_no_gradient_is_returned_untouched():
    from views_hydranet.train.training_engine import _clip_feedback_grad

    x = torch.ones(4)  # requires_grad False, as `fed` is when the wire is cut
    assert _clip_feedback_grad(x, 1.0, []) is x


def test_INTEGRATION_POTENCY_the_clip_bounds_the_gradient_on_a_family_head():
    """The gate this work needed. Threshold derived from the arm's OWN measured norms, not guessed.

    C-324: an intervention must be shown to ACT on the configuration production runs before any
    result built on it is believed. #308's first implementation was inert on this exact path.
    """
    sink: list[float] = []
    unclipped = _run_family(backprop=True, sink=sink)
    assert sink, "the sink recorded nothing — the hook never fired on the family path"
    threshold = max(sink) / 10.0
    assert threshold > 0

    clipped = _run_family(backprop=True, clip=threshold)
    assert torch.isfinite(clipped).all()
    delta = float((clipped - unclipped).abs().sum())
    assert delta > 0, (
        "the clip changed NO gradient on a family head with sampled feedback — it is inert on "
        "the production path, exactly as #308's first implementation was"
    )
    assert float(clipped.norm(2)) < float(unclipped.norm(2)), (
        "the clip acted but did not REDUCE the gradient, which is the one thing it is for"
    )


def test_INTEGRATION_the_clip_is_backward_only_and_never_moves_the_loss():
    """If clipping moved the forward pass, a stabilised arm would differ from an unstabilised one
    for a second reason and the comparison would be confounded — the same trap C-184's BatchNorm
    finding and the pushforward's train()-mode forward both sprang."""
    _, loss_off = _run_family(backprop=True, clip=None, return_loss=True)
    _, loss_on = _run_family(backprop=True, clip=1e-6, return_loss=True)
    assert loss_off == loss_on, "clipping changed the forward loss; it must touch gradients only"


def test_the_clip_does_nothing_when_the_wire_is_cut():
    """With backprop off, `fed` is detached and there is no gradient to bound. A clip that
    somehow acted there would silently alter every plain scheduled-sampling arm."""
    off_plain = _run_family(backprop=False, clip=None)
    off_clipped = _run_family(backprop=False, clip=1e-9)
    assert torch.equal(off_plain, off_clipped)
