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
# C-328 instance 5 (S3/#356): dropout is a training-only augmentation too.
#
# These assert MODULE STATE during the forward, not a call count — `model.train()`
# turns dropout on unconditionally, so a flag-only check would have passed while
# 14 of the 15 BatchNorms recomputed their statistics downstream of a live mask.
# ---------------------------------------------------------------------------
def _dropout_state_spy(monkeypatch):
    """Record, per `train()` call, the flag AND whether any dropout module was masking."""
    import torch.nn as nn

    from views_hydranet.architectures.locked_dropout import LockedDropout

    calls: list[tuple[bool, bool]] = []
    real_train = te.train

    def _rec_train(ctx, *a, **kw):
        out = real_train(ctx, *a, **kw)
        # read AFTER the call: `train()` sets model.train() then applies the gate, and the flip
        # is deliberately not restored (the next call's model.train() re-enables it).
        live = any(
            m.training
            for m in ctx.model.modules()
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, LockedDropout))
        )
        calls.append((kw.get("training_augmentation", True), live))
        return out

    monkeypatch.setattr(te, "train", _rec_train)
    return calls


def test_no_recalibration_forward_runs_with_dropout_live(monkeypatch):
    """The property, stated generally: augmentation off => no dropout module masking."""
    calls = _dropout_state_spy(monkeypatch)
    _run_loop(loop_config(random_flips=True))
    assert calls, "train() was never called"
    offending = [(flag, live) for flag, live in calls if flag is False and live]
    assert not offending, (
        f"{len(offending)} recalibration forward(s) ran with dropout masking. 14 of the 15 "
        "BatchNorms sit downstream of a dropout site, so those buffers were recomputed on "
        "activations inflated by 1/(1-p) while inference runs with dropout off (C-328)."
    )


def test_ordinary_training_calls_DO_keep_dropout_live(monkeypatch):
    """Anti-vacuity. If the gate disabled dropout everywhere, training itself would change.

    This is the test that fails if the fix is applied unconditionally instead of behind the flag.
    """
    calls = _dropout_state_spy(monkeypatch)
    _run_loop(loop_config(random_flips=True, total_lessons=8, windows_per_lesson=2))
    trained = [live for flag, live in calls if flag is True]
    assert trained, "no ordinary training call was made — the fixture proves nothing"
    assert any(trained), (
        "dropout was off on EVERY ordinary training call — the gate is too broad and training "
        "itself has changed"
    )


def test_batchnorm_still_accumulates_while_dropout_is_off(monkeypatch):
    """`model.eval()` would have been the wrong fix: it freezes the statistics the pass recomputes.

    Asserts the two halves move independently — dropout off, BatchNorm still training.
    """
    import torch.nn as nn

    from views_hydranet.architectures.locked_dropout import LockedDropout

    seen: list[tuple[bool, bool]] = []
    real_train = te.train

    def _rec(ctx, *a, **kw):
        out = real_train(ctx, *a, **kw)
        if kw.get("training_augmentation", True) is False:
            bn_live = any(
                m.training
                for m in ctx.model.modules()
                if isinstance(m, nn.modules.batchnorm._BatchNorm)
            )
            do_live = any(
                m.training
                for m in ctx.model.modules()
                if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, LockedDropout))
            )
            seen.append((bn_live, do_live))
        return out

    monkeypatch.setattr(te, "train", _rec)
    _run_loop(loop_config(random_flips=True))
    assert seen, "no recalibration pass was observed"
    assert all(bn and not do for bn, do in seen), (
        f"expected (BatchNorm training, dropout eval) on every recal pass; got {seen}"
    )


def test_a_bn_recal_run_passes_zero_epsilon(monkeypatch, tmp_path):
    """C-328 instance 5, second half: scheduled sampling is a training-only perturbation too.

    `ss_epsilon` was passed to `train()` unconditionally on the `bn_recal_from` path while
    `training_augmentation` was False — the two augmentation families gated inconsistently in the
    same call. On any SS or ITF arm the pass accumulated BatchNorm statistics while the model's own
    predictions were substituted for ground truth at rate eps, and under `ss_reverse=True` (#287)
    eps is at `ss_epsilon_max` from lesson 0.
    """
    seen: list[tuple[bool, float]] = []
    real_train = te.train

    def _rec(ctx, *a, **kw):
        seen.append((kw.get("training_augmentation", True), kw.get("ss_epsilon", 0.0)))
        return real_train(ctx, *a, **kw)

    monkeypatch.setattr(te, "train", _rec)

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

    assert seen, "train() was never called"
    offending = [(flag, eps) for flag, eps in seen if flag is False and eps != 0.0]
    assert not offending, (
        f"a recalibration pass ran with scheduled sampling live: {offending}. Those forwards "
        "re-accumulate BatchNorm statistics that ship inside the artifact."
    )


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

    # must NOT raise
    training_loop(cfg2, model2, criterion2, optimizer2, scheduler2, loop_handler(cfg2), device)
    assert state["n"] > 0, "the fixture never reached train() — the test proves nothing"


def test_an_ordinary_lesson_still_aborts_on_a_non_finite_loss(monkeypatch):
    """The other half, and the one that matters more.

    A guard that stops firing everywhere is worse than one that fires in the wrong place. ADR-014
    must still kill a real training lesson whose loss goes non-finite.
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
