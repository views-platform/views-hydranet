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

import types

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
