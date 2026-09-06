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
