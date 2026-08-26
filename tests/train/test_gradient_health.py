"""Gradient health of the real training path — the checks the suite did not have.

Audit finding (2026-08-26). Before this file:

* **No test asserted that every parameter receives gradient.** The two that came closest use
  ``any(...)`` (``test_scheduled_sampling.py``, ``test_optimization_gate.py::test_gate_18``); a
  dead regression head, a disconnected LSTM gate or a severed encoder branch passed the whole
  suite.
* **``clip_grad_norm_`` had no behavioural coverage at all.** Deleting the call at
  ``training_engine.py:1079-1080`` passed 1630 tests. ``max_norm=1.0`` is hardcoded there while
  the ``clip_grad_norm`` config field is a *bool*, so the only thing the config can say is
  on/off — and nothing checked that "on" did anything.

The clip is inline in ``training_loop``, so the only honest way to test it is to drive the real
loop and observe the gradient the optimizer is actually handed. That is what these tests do.

Measured on this vehicle at the time of writing: raw gradient norm **5.678 / 5.614** across the
two lessons, clipped norm **exactly 1.0000**. The gap is what makes the assertions discriminating.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402
from views_hydranet.train.train_model import make, training_loop  # noqa: E402
from views_hydranet.train.training_engine import _process_sequence  # noqa: E402

from .conftest import SumLoss, grad_norm, loop_config, loop_handler, make_sequence  # noqa: E402


def _norms_at_step(clip: bool):
    """Run the real ``training_loop`` and record the gradient norm as the optimizer sees it.

    The spy wraps ``optimizer.step`` rather than ``clip_grad_norm_`` deliberately: patching the
    clip would test that we call a function, and this needs to test that the gradient reaching the
    update is actually bounded. A mutation that deletes the clip call is invisible to the former
    and fatal to the latter.
    """
    cfg = loop_config(clip_grad_norm=clip)
    device = torch.device("cpu")
    torch.manual_seed(cfg["torch_seed"])
    model, criterion, optimizer, scheduler = make(cfg, device)
    handler = loop_handler(cfg)

    seen: list[float] = []
    original_step = optimizer.step

    def spy(*args, **kwargs):
        seen.append(grad_norm(model.parameters()))
        return original_step(*args, **kwargs)

    optimizer.step = spy
    training_loop(cfg, model, criterion, optimizer, scheduler, handler, device)
    return seen


def test_clip_grad_norm_true_actually_bounds_the_gradient():
    """``clip_grad_norm: True`` must leave the optimizer a gradient of norm <= 1.0.

    Kills both the "delete the clip" mutation and the "raise max_norm" mutation: the unclipped
    norm on this vehicle is ~5.6, so any max_norm above ~5.6 — or no clip at all — fails here.
    """
    norms = _norms_at_step(clip=True)
    assert norms, "optimizer.step was never called — the vehicle did not train"
    for i, n in enumerate(norms):
        assert n == pytest.approx(1.0, abs=1e-4), (
            f"lesson {i}: gradient norm at optimizer.step is {n:.4f}, not the clipped 1.0. "
            "Either clip_grad_norm_ is not being called, or its max_norm is no longer 1.0."
        )


def test_clip_grad_norm_false_leaves_the_gradient_alone():
    """The off switch must really be off, or 'clipped' and 'unclipped' arms are one run.

    Pins the measured unclipped magnitude (~5.6) well above 1.0, which is also what makes the
    test above able to tell clipping from not-clipping.
    """
    norms = _norms_at_step(clip=False)
    assert norms, "optimizer.step was never called — the vehicle did not train"
    for i, n in enumerate(norms):
        assert n > 2.0, (
            f"lesson {i}: unclipped gradient norm is {n:.4f}. The vehicle no longer produces a "
            "gradient large enough for the clip tests to discriminate; re-measure and re-tune."
        )


def test_the_balancer_log_vars_are_NOT_clipped_characterisation():
    """CHARACTERISATION of current behaviour — the clip covers ``model.parameters()`` only.

    ``MultiTaskLoss.log_vars`` is added to the optimizer as its own param group (C-111,
    ``training_engine.py:130-133``) but is not part of ``model.parameters()``, so neither the
    ``clip_grad_norm_`` call nor the ``max_raw_grad_norm`` explosion audit above it can see it.
    ``coeffs = 1 / ((is_regression + 1) * stds**2 + eps)`` has no upper bound, so an unclipped
    ``log_vars`` is an unbounded amplifier on every task loss.

    This asserts the *current* behaviour by running the exact clip call the training loop makes
    and showing a large ``log_vars`` gradient survives it untouched. Fixing this would change
    training dynamics (C-112), so the fix must break this test deliberately, not silently.
    """
    cfg = loop_config(clip_grad_norm=True, freeze_multitask_balancer=False)
    device = torch.device("cpu")
    torch.manual_seed(cfg["torch_seed"])
    model, criterion, optimizer, _scheduler = make(cfg, device)
    balancer = criterion[2]

    log_vars = next(iter(balancer.parameters()))
    assert id(log_vars) not in {id(p) for p in model.parameters()}, (
        "log_vars are now inside model.parameters(); the clip DOES cover them and this "
        "characterisation is obsolete — delete it."
    )
    assert id(log_vars) in {id(p) for g in optimizer.param_groups for p in g["params"]}, (
        "log_vars are not in the optimizer, so C-111's active balancer is not wired up"
    )

    # Give both the model and the balancer a gradient far above max_norm=1.0 ...
    for p in model.parameters():
        p.grad = torch.full_like(p, 10.0)
    log_vars.grad = torch.full_like(log_vars, 10.0)
    before = log_vars.grad.clone()

    # ... then run the clip exactly as training_engine.py:1080 runs it.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    assert grad_norm(model.parameters()) == pytest.approx(1.0, abs=1e-4), (
        "the model gradient was not clipped — the premise of this test failed"
    )
    assert torch.equal(log_vars.grad, before), (
        "log_vars.grad changed: the clip now covers the balancer. That is arguably the right "
        "behaviour, but it changes training dynamics — update this characterisation on purpose."
    )


def test_every_parameter_receives_a_finite_non_zero_gradient():
    """ALL parameters, not ``any``. A dead branch must be a test failure, not a silent null.

    Runs the real ``_process_sequence`` on the real architecture and backpropagates the real
    accumulated loss, then names every parameter that came back with no gradient, an all-zero
    gradient, or a non-finite one.
    """
    x, model, h, idx, family = make_sequence(seed=7, T=5, hw=8)
    out = _process_sequence(
        x,
        model,
        h,
        criterion_reg=FamilyLoss(family),
        criterion_class=nn.BCEWithLogitsLoss(),
        multitaskloss_instance=SumLoss(),
        idx=idx,
        device=torch.device("cpu"),
        family=family,
        ss_feedback="mean",
        forecast_composition="soft_gate",
    )
    out["total"].backward()

    missing, dead, nonfinite = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            missing.append(name)
        elif not torch.isfinite(p.grad).all():
            nonfinite.append(name)
        elif torch.count_nonzero(p.grad) == 0:
            dead.append(name)

    assert not missing, f"{len(missing)} parameters got NO gradient: {missing[:8]}"
    assert not nonfinite, f"{len(nonfinite)} parameters got a non-finite gradient: {nonfinite[:8]}"
    assert not dead, f"{len(dead)} parameters got an all-zero gradient: {dead[:8]}"


def test_the_accumulated_gradient_is_finite_and_bounded():
    """The whole-model gradient norm must be finite, and large enough to be real.

    A separate assertion from the per-parameter one: a model can have every parameter wired and
    still hand the optimizer an ``inf``. The ceiling is set two orders above the measured value
    (~2.3 on this vehicle) so it catches an explosion, not ordinary drift.
    """
    x, model, h, idx, family = make_sequence(seed=7, T=5, hw=8)
    out = _process_sequence(
        x,
        model,
        h,
        criterion_reg=FamilyLoss(family),
        criterion_class=nn.BCEWithLogitsLoss(),
        multitaskloss_instance=SumLoss(),
        idx=idx,
        device=torch.device("cpu"),
        family=family,
        ss_feedback="mean",
        forecast_composition="soft_gate",
    )
    out["total"].backward()
    n = grad_norm(model.parameters())
    assert n > 0.0, "the accumulated gradient is exactly zero — nothing trained"
    assert n == n and n != float("inf"), f"non-finite accumulated gradient norm: {n}"
    assert n < 1e3, f"accumulated gradient norm {n:.4f} looks like an explosion, not training"
