"""The ``if w_loss > 0`` gate on ``backward()`` — what it costs when the total goes negative.

``training_engine.py:1015`` guards the only ``backward()`` in the package::

    if w_loss > 0 and not _bn_recal:
        w_loss.backward()

and ``:1027`` guards the optimizer step the same way (``if lesson_loss > 0``). Read as
"skip empty windows" that is ADR-014's intent and it is correct. But ``w_loss`` comes from
``MultiTaskLoss``, whose ``forward`` returns::

    coeffs * losses + torch.log(stds + eps)

and ``log(stds)`` is **negative whenever ``log_var < 0``**. So the guard is not testing
"is there anything to learn from" — it is testing the sign of a quantity the balancer can drive
negative. When it does, the window's ``backward()`` is skipped with no warning, no log line and
no counter: the run keeps going and simply stops learning.

Measured (2026-08-26), per task, ``term(L, v) = L / ((r+1) e^v) + v/2``. Minimised over ``v`` at
``e^v = L/(r+1)``, giving ``term = 1/2 + log(L/(r+1))/2`` — **negative once ``L < (r+1)/e``**.
That is not an exotic regime; it is what a well-fitted task looks like. A balancer that converges
therefore drives the total negative and silently switches training off.

**Production is not currently exposed**: every live arm config sets
``freeze_multitask_balancer: True``, which keeps ``log_vars`` pinned at 0 (``log(stds) = 0``, all
terms non-negative). But the config default is ``False``, so the exposure is one forgotten line
away. These tests pin the mechanism and the boundary.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.train.train_model import make, training_loop  # noqa: E402
from views_hydranet.utils.mtloss import MultiTaskLoss  # noqa: E402

from .conftest import loop_config, loop_handler  # noqa: E402

#: 3 regression + 3 classification, the production task layout.
IS_REG = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

#: Per-task losses measured on the ``loop_config`` vehicle (2026-08-26), used only to place
#: ``log_vars`` at the balancer's fixed point without waiting for it to descend there.
MEASURED_TASK_LOSSES = [0.615, 0.643, 0.919, 0.0363, 0.0291, 0.0891]


def _balancer(log_vars=None):
    m = MultiTaskLoss(torch.tensor(IS_REG), reduction="sum")
    if log_vars is not None:
        with torch.no_grad():
            m.log_vars.copy_(torch.tensor(log_vars, dtype=m.log_vars.dtype))
    return m


def test_frozen_balancer_keeps_every_term_non_negative():
    """The production regime: ``log_vars == 0`` ⇒ ``log(stds) == 0`` ⇒ the guard is honest.

    This is why the finding below is latent rather than active, and it is the thing that would
    silently stop being true if the freeze flag were dropped from a config.
    """
    m = _balancer()
    assert torch.equal(m.log_vars, torch.zeros(6)), "log_vars no longer initialise to zero"
    for losses in ([1.0] * 6, [0.1] * 6, [1e-4] * 6):
        total = m(torch.tensor(losses))
        assert total >= 0.0, (
            f"frozen balancer produced a negative total {total.item()} for {losses}"
        )
    assert m(torch.zeros(6)).item() == 0.0, "an all-zero window should score exactly 0"


def test_an_active_balancer_can_drive_the_total_negative():
    """The mechanism. A well-fitted task at its own optimal ``log_var`` contributes < 0.

    Uses the analytic optimum ``v* = log(L/(r+1))`` rather than a hand-picked number, so the test
    states the property (converged balancer ⇒ negative total) instead of a magic constant.
    """
    loss = 0.05  # a well-fitted task
    opt_v = [torch.log(torch.tensor(loss / (r + 1.0))).item() for r in IS_REG]
    m = _balancer(opt_v)
    total = m(torch.full((6,), loss))
    assert total.item() < 0.0, (
        f"expected a negative total at the balancer optimum, got {total.item():.4f}. "
        "MultiTaskLoss no longer adds log(stds); re-derive this test."
    )


@pytest.mark.parametrize("loss,expect_negative", [(0.05, True), (0.2, True), (1.0, False)])
def test_the_sign_boundary_is_where_the_task_loss_falls_below_one_over_e(loss, expect_negative):
    """Pins WHERE the flip happens: a regression task's term goes negative for ``L < 2/e ≈ 0.736``.

    Recomputed from the closed form rather than asserted from a table, so a change to the
    ``(is_regression + 1)`` weighting or the ``log(stds)`` term is caught here.
    """
    v = torch.log(torch.tensor(loss / 2.0)).item()  # optimum for a regression task (r = 1)
    m = MultiTaskLoss(torch.tensor([1.0]), reduction="sum")
    with torch.no_grad():
        m.log_vars.copy_(torch.tensor([v]))
    term = m(torch.tensor([loss])).item()
    assert (term < 0.0) is expect_negative, (
        f"loss={loss}: term={term:.4f}, expected negative={expect_negative}"
    )


def _run_lessons(*, freeze: bool, lessons: int, pin_log_vars: bool, count_backwards: bool = False):
    """Drive the real ``training_loop``; count optimizer updates, or backward passes.

    ``count_backwards`` counts gradient *accumulations* on a single parameter via a
    post-accumulate-grad hook. That is the only way to observe the ``if w_loss > 0`` guard at
    ``:1015`` on its own: the ``if lesson_loss > 0`` guard at ``:1027`` independently suppresses
    the optimizer step, so counting steps cannot distinguish the two. Removing the backward guard
    alone left an earlier version of this file entirely green.
    """
    cfg = loop_config(
        clip_grad_norm=True,
        freeze_multitask_balancer=freeze,
        total_lessons=lessons,
        bn_recalibrate=False,  # the BN-recal pass is forward-only; excluded so the count is clean
    )
    device = torch.device("cpu")
    torch.manual_seed(cfg["torch_seed"])
    model, criterion, optimizer, scheduler = make(cfg, device)
    balancer = criterion[2]

    if pin_log_vars:
        # The balancer's own fixed point for this vehicle's measured task losses:
        # v* = log(L / (r + 1)). Descent moves toward here; starting here just skips the wait.
        raw = torch.tensor(MEASURED_TASK_LOSSES)
        with torch.no_grad():
            balancer.log_vars.copy_(torch.log(raw / (balancer.is_regression + 1.0)))

    events: list[int] = []
    if count_backwards:
        probe = next(p for p in model.parameters() if p.requires_grad)
        probe.register_post_accumulate_grad_hook(lambda _p: events.append(1))
    else:
        original_step = optimizer.step
        optimizer.step = lambda *a, **k: (events.append(1), original_step(*a, **k))[1]
    training_loop(cfg, model, criterion, optimizer, scheduler, loop_handler(cfg), device)
    return len(events)


def test_the_backward_guard_itself_stops_firing_not_just_the_optimizer_step():
    """Isolates the ``if w_loss > 0`` guard from the ``if lesson_loss > 0`` guard below it.

    Counts gradient accumulations rather than optimizer steps. With the frozen balancer every
    window backpropagates; with the balancer at its fixed point most windows do not, and the
    compute spent on those forward passes is discarded.

    This test exists because a mutation that deleted the backward guard entirely — leaving only
    the lesson-level guard — passed every other test in this file.
    """
    frozen = _run_lessons(freeze=True, lessons=6, pin_log_vars=False, count_backwards=True)
    active = _run_lessons(freeze=False, lessons=6, pin_log_vars=True, count_backwards=True)
    assert frozen == 6, (
        f"the frozen control backpropagated {frozen} times, expected one per lesson"
    )
    assert active < frozen, (
        f"the active balancer still backpropagated {active} times out of {frozen}: the guard at "
        "training_engine.py:1015 is no longer skipping non-positive windows."
    )


def test_the_frozen_balancer_trains_every_lesson():
    """Control: production (``freeze_multitask_balancer: True``) updates once per lesson."""
    assert _run_lessons(freeze=True, lessons=6, pin_log_vars=False) == 6, (
        "the frozen-balancer control no longer trains every lesson — the comparison below is void"
    )


def test_an_active_balancer_silently_stops_training_partway_through_the_run(caplog):
    """CHARACTERISATION, and the finding: the run keeps going, the model stops learning.

    Same vehicle, same seed, same number of lessons — only the balancer differs. With
    ``log_vars`` at the balancer's own fixed point, the total goes negative after a couple of
    lessons, ``if w_loss > 0`` stops calling ``backward()`` and ``if lesson_loss > 0`` stops
    calling ``optimizer.step()``. Measured here: **2 updates out of 6 lessons**, versus 6 of 6
    for the frozen control.

    Nothing above DEBUG is logged when this happens, which is the part that makes it dangerous:
    the progress bar advances, the loss curve is written, wandb receives rows, and the run looks
    healthy from the outside.

    Not currently reachable in production — every live arm config pins
    ``freeze_multitask_balancer: True``. But the *config default is* ``False``
    (``config_initializer.py:274``), so a new arm config that omits the line gets this regime.
    """
    lessons = 6
    with caplog.at_level(logging.DEBUG):
        updates = _run_lessons(freeze=False, lessons=lessons, pin_log_vars=True)

    assert updates < lessons, (
        f"all {lessons} lessons produced an update — the total never went non-positive, so this "
        "characterisation no longer reproduces. Re-measure MEASURED_TASK_LOSSES."
    )
    complaints = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and any(w in r.getMessage().lower() for w in ("skip", "negative", "non-positive"))
    ]
    assert not complaints, (
        f"{lessons - updates} lessons were skipped and something now warns about it — good, but "
        f"this test documented the SILENCE. Update it deliberately. Got: {complaints[:3]}"
    )
