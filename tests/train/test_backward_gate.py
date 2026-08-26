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

**Production was never exposed**: every live arm config sets ``freeze_multitask_balancer: True``,
which keeps ``log_vars`` pinned at 0 (``log(stds) = 0``, all terms non-negative). But the config
default is ``False``, so the exposure was one forgotten line away.

**FIXED (C-312).** The guards now ask the question ADR-014 means to ask — *did this window have
anything to learn from?* — rather than testing a sign. A window backpropagates unless its loss is
exactly zero; a non-finite loss raises instead of being skipped; and the optimization gate keys on
whether any window actually produced gradient. Measured on the same vehicle: **6 updates across 6
lessons**, where the old guards gave 2. Under the frozen balancer the old and new predicates select
identically, so no existing result moves — proved by
:func:`test_the_new_guard_is_byte_identical_when_frozen` rather than asserted.
"""

from __future__ import annotations

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


def test_the_backward_guard_fires_on_every_window_that_has_data():
    """C-312 regression: an active balancer must no longer suppress backward passes.

    Counts gradient accumulations rather than optimizer steps, because the optimization gate is a
    *second*, independent guard — counting steps cannot see this one. (A mutation deleting the
    backward guard passed every other test in this file until this was added.)

    Pre-fix this vehicle backpropagated on 2 of 6 windows with the balancer at its fixed point.
    """
    frozen = _run_lessons(freeze=True, lessons=6, pin_log_vars=False, count_backwards=True)
    active = _run_lessons(freeze=False, lessons=6, pin_log_vars=True, count_backwards=True)
    assert frozen == 6, (
        f"the frozen control backpropagated {frozen} times, expected one per lesson"
    )
    assert active == frozen, (
        f"the active balancer backpropagated {active} times out of {frozen}: a negative total is "
        "suppressing backward again (C-312)."
    )


def test_an_active_balancer_no_longer_stops_training_partway_through_the_run():
    """C-312 regression, at the level that matters: every lesson still updates the weights.

    Same vehicle, same seed, only the balancer differs. Pre-fix: 2 updates across 6 lessons, with
    nothing logged — the run looked healthy from the outside while the model had stopped learning.
    """
    lessons = 6
    updates = _run_lessons(freeze=False, lessons=lessons, pin_log_vars=True)
    assert updates == lessons, (
        f"only {updates} of {lessons} lessons produced an update. The optimization gate is keying "
        "on the sign of the accumulated loss again (C-312)."
    )


def test_the_new_guard_is_byte_identical_when_frozen():
    """The C-112 comparability proof: under production config the two predicates cannot differ.

    C-112 records that changing training dynamics makes pre/post-fix model metrics incomparable,
    so this fix is only safe if it is a no-op on the configuration every existing result was
    produced under. With ``freeze_multitask_balancer: True`` the balancer contributes
    ``log(stds) = 0`` and every task term is non-negative, so ``w_loss >= 0`` always and the old
    ``> 0`` and new ``!= 0`` select the same windows.

    Asserted by observing every window's total during a real run rather than by re-deriving the
    algebra, so it stays true if the loss composition changes.
    """
    cfg = loop_config(
        clip_grad_norm=True, freeze_multitask_balancer=True, total_lessons=4, bn_recalibrate=False
    )
    device = torch.device("cpu")
    torch.manual_seed(cfg["torch_seed"])
    model, criterion, optimizer, scheduler = make(cfg, device)

    totals: list[float] = []
    original_train = training_loop.__globals__["train"]

    def spy(ctx, handler, pbar, **kwargs):
        result = original_train(ctx, handler, pbar, **kwargs)
        totals.append(result["total"].item())
        return result

    training_loop.__globals__["train"] = spy
    try:
        training_loop(cfg, model, criterion, optimizer, scheduler, loop_handler(cfg), device)
    finally:
        training_loop.__globals__["train"] = original_train

    assert totals, "no windows ran — the vehicle did not train"
    assert all(t >= 0.0 for t in totals), (
        f"a frozen-balancer window scored below zero (min {min(totals):.6f}). The equal-weighting "
        "regime no longer guarantees a non-negative total, so this fix is NOT a no-op on the "
        "configuration every existing result was produced under — stop and re-derive."
    )
    disagreements = [t for t in totals if (t > 0.0) != (t != 0.0)]
    assert not disagreements, (
        f"old predicate (> 0) and new predicate (!= 0) disagree on {len(disagreements)} windows: "
        f"{disagreements[:3]}"
    )


def test_an_empty_window_still_skips_backward():
    """ADR-014's actual intent survives the fix: a zero loss means no supervised cells, so no step.

    Guards against 'fixing' C-312 by simply always backpropagating, which would spend a full
    backward pass on windows with nothing in them.
    """
    cfg = loop_config(freeze_multitask_balancer=True, total_lessons=1, bn_recalibrate=False)
    device = torch.device("cpu")
    torch.manual_seed(cfg["torch_seed"])
    model, criterion, optimizer, scheduler = make(cfg, device)

    original_train = training_loop.__globals__["train"]

    def zero_loss(ctx, handler, pbar, **kwargs):
        result = original_train(ctx, handler, pbar, **kwargs)
        result["total"] = torch.zeros((), requires_grad=True)
        return result

    training_loop.__globals__["train"] = zero_loss
    steps: list[int] = []
    original_step = optimizer.step
    optimizer.step = lambda *a, **k: (steps.append(1), original_step(*a, **k))[1]
    try:
        training_loop(cfg, model, criterion, optimizer, scheduler, loop_handler(cfg), device)
    finally:
        training_loop.__globals__["train"] = original_train

    assert not steps, "an all-zero (empty) window still triggered an optimizer step"


def test_a_non_finite_loss_raises_instead_of_being_skipped():
    """A NaN loss used to fail ``w_loss > 0`` and be silently skipped like an empty window.

    Silently discarding a NaN window hides the upstream bug that produced it — C-212 was exactly
    such a bug — so the fix raises rather than skipping. ``RuntimeError`` to match
    ``IntegrityGuardian.monitor``, which raises on the same condition twenty lines further down.
    """
    cfg = loop_config(freeze_multitask_balancer=True, total_lessons=1, bn_recalibrate=False)
    device = torch.device("cpu")
    torch.manual_seed(cfg["torch_seed"])
    model, criterion, optimizer, scheduler = make(cfg, device)

    original_train = training_loop.__globals__["train"]

    def nan_loss(ctx, handler, pbar, **kwargs):
        result = original_train(ctx, handler, pbar, **kwargs)
        result["total"] = torch.tensor(float("nan"), requires_grad=True)
        return result

    training_loop.__globals__["train"] = nan_loss
    try:
        with pytest.raises(RuntimeError, match="not a finite number"):
            training_loop(cfg, model, criterion, optimizer, scheduler, loop_handler(cfg), device)
    finally:
        training_loop.__globals__["train"] = original_train
