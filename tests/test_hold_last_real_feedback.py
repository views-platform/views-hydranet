"""PROBE-A (#324): `hold_last_real` feeds the ORIGIN's month at every step, never the model's own.

This transform exists to bound the prize of the direct-multi-horizon epic before it is built.
Paired with `freeze_recurrent='cell'` it reproduces direct multi-horizon's **inference** semantics
on an already-trained artifact: the state is held, the input never evolves, and nothing the model
emits is ever fed back.

The property that makes it a valid probe is the one tested first: the source month must be the
**origin**, constant across every step — not `step`, which is `use_real`, and not an offset, which
is `wrong_month`. A transform that silently tracked `step` would be teacher forcing wearing a
different name, and the probe would read as a spectacular win that means nothing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.utils.feedback_field_transforms import (  # noqa: E402
    FEEDBACK_TRANSFORMS,
    parse_feedback_transform,
)


def test_the_transform_is_registered_and_takes_no_parameter():
    assert "hold_last_real" in FEEDBACK_TRANSFORMS
    assert FEEDBACK_TRANSFORMS["hold_last_real"] is False, (
        "hold_last_real takes no parameter; True would demand one and change the spec"
    )
    assert parse_feedback_transform("hold_last_real") == ("hold_last_real", None)


class _Stub:
    """Minimal stand-in exercising only `_apply_feedback_transform`'s step-remapping branch."""

    _month_shuffle: dict[int, int] = {}

    def __init__(self, arm):
        self._feedback_arm = arm

    def _real_dynamic(self, full_tensor, model_in_indices, n_dyn, step):
        from views_hydranet.utils.hydranet_inference import HydraNetInference

        return HydraNetInference._real_dynamic(self, full_tensor, model_in_indices, n_dyn, step)

    def apply(self, t0, full, idx, n_static, step, origin):
        from views_hydranet.utils.hydranet_inference import HydraNetInference

        return HydraNetInference._apply_feedback_transform(
            self, t0, full, idx, n_static, step, origin
        )


def _fixture():
    """A window whose month `m` is filled with the constant `m` — so the source month is readable
    straight off the returned tensor."""
    B, M, C, H, W = 1, 12, 3, 4, 4
    full = torch.zeros(B, M, C, H, W)
    for m in range(M):
        full[:, m] = float(m)
    idx = [0, 1]  # 2 dynamic channels; channel 2 stands in for a static
    t0 = torch.full((B, 3, H, W), -1.0)  # the "model's own" field: a value no month carries
    return full, idx, t0


@pytest.mark.parametrize("step", [4, 5, 9, 11])
def test_hold_last_real_returns_the_ORIGIN_month_at_every_step(step):
    """The load-bearing property. Origin is fixed at 3; the answer must be 3 for every step."""
    full, idx, t0 = _fixture()
    out = _Stub(("hold_last_real", None)).apply(t0, full, idx, 1, step=step, origin=3)
    assert torch.allclose(out[:, :2], torch.full_like(out[:, :2], 3.0)), (
        f"step={step}: fed field came from month {out[0, 0, 0, 0].item()}, not origin 3"
    )


def test_use_real_still_tracks_step_so_the_two_are_not_the_same_transform():
    """Anti-vacuity: if `use_real` also returned a constant, the test above would prove nothing."""
    full, idx, t0 = _fixture()
    seen = {
        s: _Stub(("use_real", None)).apply(t0, full, idx, 1, step=s, origin=3)[0, 0, 0, 0].item()
        for s in (4, 5, 9)
    }
    assert seen == {4: 4.0, 5: 5.0, 9: 9.0}, f"use_real stopped tracking step: {seen}"


def test_the_model_field_is_fully_replaced_not_blended():
    """The probe's premise is that NOTHING the model emitted is fed back."""
    full, idx, t0 = _fixture()
    out = _Stub(("hold_last_real", None)).apply(t0, full, idx, 1, step=7, origin=2)
    assert not (out[:, :2] == -1.0).any(), "the model's own field survived into the fed input"


def test_statics_are_untouched():
    full, idx, t0 = _fixture()
    out = _Stub(("hold_last_real", None)).apply(t0, full, idx, 1, step=7, origin=2)
    assert torch.allclose(out[:, 2:], t0[:, 2:]), "a step remapping must not disturb statics"
