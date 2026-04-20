"""
Tests for WarmupDecayLearningRateScheduler.

First introduction of @pytest.mark.parametrize into the suite.
Green/Beige/Red taxonomy (ADR-005).

Note: PyTorch _LRScheduler.__init__ calls step() internally, so after
construction last_epoch=0 and step_num=1 has already been evaluated.
Each subsequent scheduler.step() advances by one more.
"""

import math

import pytest
import torch
import torch.nn as nn

from views_hydranet.utils.warmup_decay_lr_scheduler import WarmupDecayLearningRateScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_scheduler(d=512, warmup_steps=4000, lr=1.0, n_param_groups=1):
    """Factory: build optimizer + scheduler with given hyperparams."""
    layers = [nn.Linear(1, 1) for _ in range(n_param_groups)]
    param_groups = [{"params": layer.parameters(), "lr": lr} for layer in layers]
    optimizer = torch.optim.SGD(param_groups)
    scheduler = WarmupDecayLearningRateScheduler(optimizer, d=d, warmup_steps=warmup_steps)
    return scheduler


def _get_lr_at_epoch(d, warmup_steps, epoch):
    """Advance a fresh scheduler `epoch` additional steps beyond construction, return lr.

    After construction, last_epoch=0 (step_num=1).
    After `epoch` calls to step(), last_epoch=epoch (step_num=epoch+1).
    """
    sched = _make_scheduler(d=d, warmup_steps=warmup_steps)
    for _ in range(epoch):
        sched.step()
    return sched.get_lr()[0]


def _expected_lr(d, warmup_steps, step_num):
    """Reference formula computed independently from step_num (1-based)."""
    scale = d ** (-0.5)
    return scale * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))


# ---------------------------------------------------------------------------
# GREEN TEAM — happy path
# ---------------------------------------------------------------------------
class TestGreen:
    def test_green_lr_at_construction(self):
        """After construction, step_num=1 is already computed."""
        sched = _make_scheduler(d=512, warmup_steps=4000)
        lr = sched.get_lr()[0]
        expected = _expected_lr(512, 4000, step_num=1)
        assert math.isclose(lr, expected, rel_tol=1e-9)

    def test_green_lr_increases_during_warmup(self):
        """Early steps with warmup=100: monotonically increasing."""
        sched = _make_scheduler(d=128, warmup_steps=100)
        lrs = [sched.get_lr()[0]]
        for _ in range(9):
            sched.step()
            lrs.append(sched.get_lr()[0])
        assert all(lrs[i] < lrs[i + 1] for i in range(len(lrs) - 1))

    def test_green_lr_decreases_after_warmup(self):
        """Steps well past warmup: monotonically decreasing."""
        sched = _make_scheduler(d=128, warmup_steps=10)
        # Advance well past warmup
        for _ in range(20):
            sched.step()
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(sched.get_lr()[0])
        assert all(lrs[i] > lrs[i + 1] for i in range(len(lrs) - 1))

    def test_green_peak_near_warmup_step(self):
        """LR peaks near the warmup step."""
        warmup = 50
        sched = _make_scheduler(d=64, warmup_steps=warmup)
        lrs = [sched.get_lr()[0]]
        for _ in range(warmup + 20):
            sched.step()
            lrs.append(sched.get_lr()[0])
        peak_idx = lrs.index(max(lrs))
        # Peak should be within 2 of warmup (off-by-one from init step)
        assert abs(peak_idx - warmup) <= 2


# ---------------------------------------------------------------------------
# BEIGE TEAM — boundary & robustness (parametrized)
# ---------------------------------------------------------------------------
class TestBeige:
    @pytest.mark.parametrize("extra_steps", [0, 9, 99, 999, 9999])
    def test_beige_lr_positive_all_steps(self, extra_steps):
        """LR must be positive at every step."""
        lr = _get_lr_at_epoch(d=512, warmup_steps=4000, epoch=extra_steps)
        assert lr > 0

    @pytest.mark.parametrize("d,warmup", [(64, 100), (128, 500), (512, 4000)])
    def test_beige_formula_parity(self, d, warmup):
        """Scheduler output matches independently computed formula at multiple epochs."""
        for epoch in [0, warmup // 2, warmup, warmup * 2]:
            lr = _get_lr_at_epoch(d=d, warmup_steps=warmup, epoch=epoch)
            step_num = epoch + 1
            expected = _expected_lr(d, warmup, step_num)
            assert math.isclose(lr, expected, rel_tol=1e-9), (
                f"Mismatch at epoch={epoch} (step_num={step_num}): got {lr}, expected {expected}"
            )

    def test_beige_multiple_param_groups(self):
        """Returns list matching param group count."""
        sched = _make_scheduler(d=512, warmup_steps=4000, n_param_groups=3)
        lrs = sched.get_lr()
        assert len(lrs) == 3
        assert all(lr == lrs[0] for lr in lrs)


# ---------------------------------------------------------------------------
# RED TEAM — failure detection
# ---------------------------------------------------------------------------
class TestRed:
    def test_red_warmup_zero_raises_during_init(self):
        """warmup_steps=0 -> ZeroDivisionError during construction."""
        with pytest.raises(ZeroDivisionError):
            _make_scheduler(d=512, warmup_steps=0)

    def test_red_negative_d_produces_complex(self):
        """d=-1 -> complex lr ((-1)^(-0.5) is imaginary). Documents edge case."""
        sched = _make_scheduler(d=-1, warmup_steps=100)
        lr = sched.get_lr()[0]
        assert isinstance(lr, complex)
