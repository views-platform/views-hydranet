"""BN recalibration passes must see CLEAN data — every augmentation, not only the noise.

C-328 instance 4. `_recalibrate_bn` (and a `bn_recal_from` run) recompute BatchNorm running
statistics, which are **saved into the artifact and used at inference**. `train()` has always
flipped the tube under `random_flips` (default **True**), and `_recalibrate_bn` calls `train()` —
so with `momentum=None` (equal cumulative weighting) roughly half of every BatchNorm statistic in
every shipped artifact was accumulated on H/W-flipped fields. Convolution is not flip-equivariant,
so those buffers were biased toward a distribution inference never produces, partly defeating the
fix C-184 exists to be.

The first fix for C-328 suppressed the *input noise* on one of the two recalibration paths and left
the flip — older than the noise — untouched on both. These tests pin the general property instead
of the instance: **no training-only augmentation reaches a recalibration pass.**
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import views_hydranet.train.training_engine as te  # noqa: E402
from views_hydranet.train.training_engine import make, train, training_loop  # noqa: E402

from .conftest import loop_config, loop_handler  # noqa: E402


def _paired_spy(monkeypatch):
    """Record, per `train()` call, the augmentation flag AND whether the tube was flipped.

    Pairing them is the point: it is not enough to know the flag was passed, nor that some flip
    happened somewhere. What must hold is that **no call with the flag off flipped**.
    """
    from views_hydranet.utils.volume_handler import VolumeHandler

    calls: list[tuple[bool, int]] = []
    real_train, real_flip = te.train, VolumeHandler.flip
    n_flips = {"n": 0}

    def _rec_flip(self, axis):
        n_flips["n"] += 1
        return real_flip(self, axis)

    def _rec_train(*a, **kw):
        before = n_flips["n"]
        out = real_train(*a, **kw)
        calls.append((kw.get("training_augmentation", True), n_flips["n"] - before))
        return out

    monkeypatch.setattr(VolumeHandler, "flip", _rec_flip)
    monkeypatch.setattr(te, "train", _rec_train)
    return calls


def _run_loop(cfg):
    device = torch.device("cpu")
    torch.manual_seed(cfg["torch_seed"])
    model, criterion, optimizer, scheduler = make(cfg, device)
    training_loop(cfg, model, criterion, optimizer, scheduler, loop_handler(cfg), device)


def test_no_call_with_augmentation_OFF_ever_flips(monkeypatch):
    """The behavioural core of C-328 instance 4, through the real loop."""
    calls = _paired_spy(monkeypatch)
    _run_loop(loop_config(random_flips=True))
    suppressed = [(flag, n) for flag, n in calls if flag is False]
    assert suppressed, "no suppressed call was made — the BN recalibration pass did not run"
    assert all(n == 0 for _, n in suppressed), (
        f"a recalibration pass flipped the tube: {suppressed}. Those buffers ship in the artifact "
        "and are used at inference, where the field is never flipped."
    )


def test_the_fixture_DOES_flip_on_ordinary_training_calls(monkeypatch):
    """Anti-vacuity for the test above: if this fixture never flips at all, that test proves
    nothing. Round 3 found two assertions that could not fail; this is the paired positive case
    that stops this one joining them. `random_flips` is a coin flip per axis per window, so this
    accumulates across the loop's windows rather than relying on one draw."""
    calls = _paired_spy(monkeypatch)
    _run_loop(loop_config(random_flips=True, total_lessons=8, windows_per_lesson=2))
    augmented = [(flag, n) for flag, n in calls if flag is True]
    assert augmented, "no ordinary training call was made"
    assert sum(n for _, n in augmented) > 0, (
        "the fixture never flipped even with augmentation ON, so the off-test is vacuous"
    )


def _train_spy(monkeypatch):
    calls: list[bool] = []
    real = te.train

    def _rec(*a, **kw):
        calls.append(kw.get("training_augmentation", True))
        return real(*a, **kw)

    monkeypatch.setattr(te, "train", _rec)
    return calls


def test_recalibrate_bn_suppresses_augmentation(monkeypatch):
    """`_recalibrate_bn` must pass the flag; the in-process recalibration path."""
    calls = _train_spy(monkeypatch)
    cfg = loop_config(random_flips=True)
    device = torch.device("cpu")
    model, criterion, optimizer, scheduler = make(cfg, device)
    training_loop(cfg, model, criterion, optimizer, scheduler, loop_handler(cfg), device)
    assert calls, "train() was never called"
    assert any(c is True for c in calls), "no ordinary training call — the fixture is wrong"
    assert any(c is False for c in calls), (
        "no call suppressed augmentation — the BN recalibration pass ran with flips and noise live"
    )


def test_a_bn_recal_from_run_suppresses_augmentation(monkeypatch, tmp_path):
    """The second recalibration path: `bn_recal_from` drives the NORMAL lesson loop forward-only.

    It leaves autograd ENABLED, which is why a grad-state assertion cannot reach it — the reason
    the first fix for C-328 missed it.
    """
    calls = _train_spy(monkeypatch)
    cfg = loop_config(random_flips=True)
    device = torch.device("cpu")
    model, criterion, optimizer, scheduler = make(cfg, device)
    ckpt = tmp_path / "arm.pt"
    torch.save(model.state_dict(), ckpt)

    cfg2 = loop_config(random_flips=True, bn_recal_from=str(ckpt))
    model2, criterion2, optimizer2, scheduler2 = make(cfg2, device)
    calls.clear()
    training_loop(cfg2, model2, criterion2, optimizer2, scheduler2, loop_handler(cfg2), device)
    assert calls, "train() was never called"
    assert all(c is False for c in calls), (
        f"a bn_recal_from run made an augmented forward: {calls}. Every pass in that run "
        "re-accumulates BatchNorm statistics that ship in the artifact."
    )


def test_the_flag_defaults_to_augmenting():
    """`training_augmentation` must default **True** — the value that leaves training alone.

    Every call site in the repo passes it explicitly, so this default is only reached by a
    caller added later. That is exactly the caller C-328 is about: *adding a path through the
    training input transform without asking what it does*. Defaulting False would silently
    disable both augmentations for any such caller, and no other test in the suite would fail
    — a mutation that survived the audit until this test existed.
    """
    import inspect

    default = inspect.signature(train).parameters["training_augmentation"].default
    assert default is True, f"training_augmentation defaults to {default!r}, not True"


# ---------------------------------------------------------------------------
# C-328 instance 5 (S3/#356): scheduled sampling is a training-only perturbation too.
#
# Dropout is deliberately NOT in this gate. An earlier draft of this story switched every
# dropout module to eval() on the recal passes, on the premise that inference runs dropout off.
# It does not: inference is `model.eval()` + `set_locked_dropout(True)` (ADR-057, MC-dropout),
# so BN statistics must be estimated WITH dropout on. `/code-review max` on #372 measured the
# draft's effect — running_var 14-21% low at 14 of 15 BN layers — and it was reverted.
# ---------------------------------------------------------------------------
def _dropout_state_during_forward_spy(monkeypatch):
    """Record, per `train()` call, the flag and the set of dropout `.training` states observed
    AT FORWARD TIME by a pre-hook — not after the call returns. An after-return read cannot tell
    "off during the BN-accumulating forward" from "flipped afterwards" (a mutation that moved the
    gate to just before `train_log` passed the earlier after-return version of this spy)."""
    import torch.nn as nn

    from views_hydranet.architectures.locked_dropout import LockedDropout

    calls: list[tuple[bool, set[bool]]] = []
    real_train = te.train
    hooked: set[int] = set()

    def _rec_train(ctx, *a, **kw):
        seen: set[bool] = set()

        def _hook(m, _inp):
            seen.add(m.training)

        handles = []
        for m in ctx.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, LockedDropout)):
                handles.append(m.register_forward_pre_hook(_hook))
                hooked.add(id(m))
        try:
            out = real_train(ctx, *a, **kw)
        finally:
            for h in handles:
                h.remove()
        calls.append((kw.get("training_augmentation", True), seen))
        return out

    monkeypatch.setattr(te, "train", _rec_train)
    return calls, hooked


def test_recalibration_forwards_keep_dropout_live_like_inference_does(monkeypatch):
    """Pins the REVERT. Dropout modules are in train mode during every recal forward, because
    that is the regime inference runs (ADR-057) and the regime the C-184 fix was validated in."""
    calls, hooked = _dropout_state_during_forward_spy(monkeypatch)
    _run_loop(loop_config(random_flips=True))
    assert hooked, "the model has no dropout modules — this test observes nothing"
    recal = [seen for flag, seen in calls if flag is False]
    assert recal, "no recalibration forward was observed"
    assert all(seen == {True} for seen in recal), (
        f"dropout state during recal forwards: {recal}. It must be train mode throughout: "
        "inference runs MC-dropout ON (set_locked_dropout(True), ADR-057), so BatchNorm must "
        "normalise dropout-shaped activations. Switching it off here puts running_var 14-21% "
        "low at 14 of 15 BN layers — measured on #372 before the flip was reverted."
    )


def test_a_bn_recal_run_passes_zero_epsilon(monkeypatch, tmp_path):
    """C-328 instance 5, second half: scheduled sampling is a training-only perturbation too.

    `ss_epsilon` was passed to `train()` unconditionally on the `bn_recal_from` path while
    `training_augmentation` was False — the two augmentation families gated inconsistently in the
    same call. On any SS or ITF arm the pass accumulated BatchNorm statistics while the model's own
    predictions were substituted for ground truth at rate eps, and under `ss_reverse=True` (#287)
    eps is at `ss_epsilon_max` from lesson 0.
    """
    # Observed where it is CONSUMED (`_process_sequence`), not at the call-site kwarg: the first
    # version of this test read `kw["ss_epsilon"]` off `train()`, which cannot see a `train()`
    # that forwards a non-zero epsilon it was handed by some other caller. The flag is read off
    # `train()` and paired with the epsilon `_process_sequence` actually received inside it.
    seen: list[tuple[bool, float]] = []
    real_train, real_ps = te.train, te._process_sequence
    current = {"flag": True}

    def _rec_train(ctx, *a, **kw):
        current["flag"] = kw.get("training_augmentation", True)
        return real_train(ctx, *a, **kw)

    def _rec_ps(*a, **kw):
        seen.append((current["flag"], float(kw.get("ss_epsilon", 0.0))))
        return real_ps(*a, **kw)

    monkeypatch.setattr(te, "train", _rec_train)
    monkeypatch.setattr(te, "_process_sequence", _rec_ps)

    device = torch.device("cpu")
    cfg = loop_config(random_flips=True)
    model, criterion, optimizer, scheduler = make(cfg, device)
    ckpt = tmp_path / "arm.pt"
    torch.save(model.state_dict(), ckpt)

    # scheduled sampling ACTIVE, so a non-zero epsilon would otherwise reach the recal pass
    cfg2 = loop_config(
        random_flips=True,
        bn_recal_from=str(ckpt),
        ss_schedule="linear",
        ss_warmup_lessons=1,
        ss_epsilon_max=1.0,
    )
    model2, criterion2, optimizer2, scheduler2 = make(cfg2, device)
    seen.clear()
    training_loop(cfg2, model2, criterion2, optimizer2, scheduler2, loop_handler(cfg2), device)

    assert seen, "_process_sequence was never called"
    assert any(flag is False for flag, _ in seen), "no recalibration forward was observed"
    offending = [(flag, eps) for flag, eps in seen if flag is False and eps != 0.0]
    assert not offending, (
        f"a recalibration pass ran with scheduled sampling live: {offending}. Those forwards "
        "re-accumulate BatchNorm statistics that ship inside the artifact."
    )


def test_train_itself_zeroes_epsilon_when_augmentation_is_off(monkeypatch):
    """`training_augmentation=False` must mean what it says INSIDE `train()`.

    The call site zeroes epsilon on a recal pass, but a comment saying the flag "gates EVERY
    training-only augmentation" was, for scheduled sampling, true only of that one call site: hand
    `train()` a live epsilon with the flag off and `_process_sequence` received it unchanged
    (C-303 shape, found by `/code-review max` on #372). Now it is gated where it is consumed.
    """
    received: list[float] = []
    real_ps = te._process_sequence

    def _rec_ps(*a, **kw):
        received.append(float(kw.get("ss_epsilon", 0.0)))
        return real_ps(*a, **kw)

    monkeypatch.setattr(te, "_process_sequence", _rec_ps)

    device = torch.device("cpu")
    cfg = loop_config(random_flips=True)
    model, criterion, optimizer, scheduler = make(cfg, device)
    criterion_reg, criterion_class, multitask = criterion  # the tuple `make` returns
    ctx = te.TrainingContext(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion_reg=criterion_reg,
        criterion_class=criterion_class,
        multitaskloss_instance=multitask,
        config=cfg,
        device=device,
        viz=None,
        forensics=None,
    )
    handler = loop_handler(cfg)
    sampler = te.VolumeSampler(handler, cfg)
    target, threshold = te.CurriculumLearner(cfg, handler).get_lesson(0)
    batch, _ = sampler.get_batch(target, threshold, batch_size=1)

    train(ctx, batch[0], None, stage_label="", training_augmentation=False, ss_epsilon=1.0)
    assert received == [0.0], (
        f"train(training_augmentation=False, ss_epsilon=1.0) passed {received} on to "
        "_process_sequence — the flag does not gate scheduled sampling inside train()"
    )

    # Anti-vacuity: with augmentation ON the epsilon must pass through. This fixture's features
    # are not SS-shaped (features != regression_targets, C-260), so the forward is stopped right
    # after the kwarg is observed rather than run.
    class _Observed(Exception):
        pass

    def _rec_then_stop(*a, **kw):
        received.append(float(kw.get("ss_epsilon", 0.0)))
        raise _Observed

    monkeypatch.setattr(te, "_process_sequence", _rec_then_stop)
    received.clear()
    with pytest.raises(_Observed):
        train(ctx, batch[0], None, stage_label="", training_augmentation=True, ss_epsilon=1.0)
    assert received == [1.0], "with augmentation ON the epsilon must reach _process_sequence"


# ---------------------------------------------------------------------------
# S4/#357: the ADR-014 integrity gate must not abort a forward-only pass.
# ---------------------------------------------------------------------------
def test_a_recal_pass_survives_a_non_finite_loss(monkeypatch, tmp_path):
    """A recalibration pass consumes no loss and computes no gradient.

    `lesson_loss` accumulates unconditionally, so one non-finite window makes it NaN, and
    `IntegrityGuardian.monitor` would then kill the pass — throwing away the corrected BatchNorm
    buffers it exists to produce. The forward-only branch already promises this does not happen
    ("behaves exactly as it did before this fix"); the C-312 gate change had quietly broken it.
    """
    device = torch.device("cpu")
    cfg = loop_config(random_flips=False)
    model, criterion, optimizer, scheduler = make(cfg, device)
    ckpt = tmp_path / "arm.pt"
    torch.save(model.state_dict(), ckpt)

    cfg2 = loop_config(random_flips=False, bn_recal_from=str(ckpt))
    model2, criterion2, optimizer2, scheduler2 = make(cfg2, device)

    # make the very first window's loss non-finite
    real_train = te.train
    state = {"n": 0}

    def _nan_once(ctx, *a, **kw):
        out = real_train(ctx, *a, **kw)
        state["n"] += 1
        if state["n"] == 1:
            out = dict(out)
            out["total"] = torch.tensor(float("nan"))
        return out

    monkeypatch.setattr(te, "train", _nan_once)

    # must NOT raise: the loss is not consumed, and the BN buffers (which are) stayed finite
    training_loop(cfg2, model2, criterion2, optimizer2, scheduler2, loop_handler(cfg2), device)
    assert state["n"] > 0, "the fixture never reached train() — the test proves nothing"


def _poison_first_batchnorm_input_once(model):
    """A forward pre-hook that feeds ONE NaN into the first BatchNorm on its first call.

    This is the case the loss-based test above cannot see: the loss injection happens AFTER a
    finite forward, so the buffers are never touched. Here the buffers themselves are poisoned,
    and with momentum=None (cumulative average) they never recover.
    """
    import torch.nn as nn

    first = next(m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    fired = {"n": 0}

    def _hook(_m, inp):
        if fired["n"] == 0:
            fired["n"] += 1
            x = inp[0].clone()
            x.view(-1)[0] = float("nan")
            return (x,) + tuple(inp[1:])

    first.register_forward_pre_hook(_hook)
    return fired


def test_a_bn_recal_from_run_refuses_to_hand_back_non_finite_buffers(monkeypatch, tmp_path):
    """S4 removed the only loud failure on this path, and nothing checked what the path actually
    produces. `/code-review max` on #372 ran it end-to-end: one NaN on one forward, every BN
    buffer NaN, `final_loss` finite, HEALTHY in the summary, `torch.save` writes it, first loud
    failure at inference. Now the run raises before it returns, naming the buffers."""
    device = torch.device("cpu")
    cfg = loop_config(random_flips=False)
    model, criterion, optimizer, scheduler = make(cfg, device)
    ckpt = tmp_path / "arm.pt"
    torch.save(model.state_dict(), ckpt)

    cfg2 = loop_config(random_flips=False, bn_recal_from=str(ckpt))
    model2, criterion2, optimizer2, scheduler2 = make(cfg2, device)
    fired = _poison_first_batchnorm_input_once(model2)

    with pytest.raises(RuntimeError, match="non-finite running statistics"):
        training_loop(cfg2, model2, criterion2, optimizer2, scheduler2, loop_handler(cfg2), device)
    assert fired["n"] == 1, "the poison hook never fired — the test proves nothing"


def test_post_training_recalibration_restores_the_trained_buffers_on_poison(monkeypatch):
    """The other recal path (`_recalibrate_bn`, default ON) sits inside a fail-safe: a raise
    there restores the pre-recal buffers and saves the trained model as-is. The finiteness
    check must raise INSIDE that fail-safe, so the outcome is 'trained buffers kept', not
    'NaN buffers shipped' and not 'training lost'."""
    import torch.nn as nn

    device = torch.device("cpu")
    cfg = loop_config(random_flips=False, bn_recalibrate=True)
    model, criterion, optimizer, scheduler = make(cfg, device)
    fired = _poison_first_batchnorm_input_once(model)
    # the hook must fire during the RECAL pass, not during training: arm it only once training
    # has finished, by keying on the reset the recal pass performs first.
    real_reset = te._reset_bn_stats

    def _arm_then_reset(m):
        fired["n"] = 0
        return real_reset(m)

    monkeypatch.setattr(te, "_reset_bn_stats", _arm_then_reset)
    fired["n"] = 1  # disarmed during training

    training_loop(cfg, model, criterion, optimizer, scheduler, loop_handler(cfg), device)

    for name, m in model.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            assert torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all(), (
                f"{name} left with non-finite running stats after a poisoned recal pass — the "
                "fail-safe did not restore the trained buffers"
            )


def test_an_ordinary_lesson_still_aborts_on_a_non_finite_loss(monkeypatch):
    """The other half. A guard that stops firing everywhere is worse than one that fires in the
    wrong place: an ordinary training lesson whose loss goes non-finite must still abort.

    Honest scope: what fires first here is the per-window `isfinite(w_loss)` check, not
    `IntegrityGuardian.monitor` — replacing the monitor call with `if False:` leaves this test
    green (measured on #372). This pins "an ordinary lesson aborts", not "the monitor fires".
    """
    device = torch.device("cpu")
    cfg = loop_config(random_flips=False)
    model, criterion, optimizer, scheduler = make(cfg, device)

    real_train = te.train

    def _nan(ctx, *a, **kw):
        out = dict(real_train(ctx, *a, **kw))
        out["total"] = torch.tensor(float("nan"))
        return out

    monkeypatch.setattr(te, "train", _nan)

    with pytest.raises(RuntimeError):
        training_loop(cfg, model, criterion, optimizer, scheduler, loop_handler(cfg), device)
