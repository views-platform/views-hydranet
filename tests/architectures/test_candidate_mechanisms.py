"""Each candidate must actually DO the thing it exists to test.

`test_bakeoff_candidates.py` checks the six architectures are well-formed — shapes,
finiteness, gradients. A `/falsify guard` audit showed that is not enough: **every** candidate's
distinguishing
mechanism could be removed, inverted or neutered and the whole suite stayed green. `WideMemory` at
factor 1 is byte-identical to its own control; `MaxBlurPool` with the blur deleted is a plain
max-pool; `DualStream`'s parallel stream multiplied by zero still has finite gradients. Each
of those would run for ~2 GPU-hours, score cleanly, and report a null that reads as *"this
architecture does
not help"* rather than *"this architecture was never applied"*.

These tests pin the mechanism itself, so a neutered candidate fails here in milliseconds instead of
costing an arm and producing a false negative.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from views_hydranet.architectures.registry import get_architecture  # noqa: E402

IN_CH, HID, H, W = 3, 32, 32, 32


def _build(name, seed=7, hidden=HID):
    torch.manual_seed(seed)
    return get_architecture(name)(IN_CH, hidden, 1, 0.0, output_distribution="nb").float().eval()


# ── (1) AntiAliasedPool: the blur must exist, be applied, and cover BOTH poolings ────────────


def test_antialiased_pool_replaces_both_poolings():
    """Blurring only `pool0` would leave the second downsampling aliasing exactly as before."""
    from views_hydranet.architectures.anti_aliased_pool import MaxBlurPool2d

    m = _build("AntiAliasedPool")
    assert isinstance(m.pool0, MaxBlurPool2d), "pool0 is not blurred"
    assert isinstance(m.pool1, MaxBlurPool2d), "pool1 is not blurred — aliasing survives"


def test_maxblurpool_output_differs_from_plain_maxpool():
    """The blur must CHANGE the result, not merely be registered as a buffer.

    A buffer can be present and never used. This is the difference between the mechanism existing
    and the mechanism being applied.
    """
    from views_hydranet.architectures.anti_aliased_pool import MaxBlurPool2d

    torch.manual_seed(0)
    x = torch.randn(1, 8, 16, 16)
    blurred = MaxBlurPool2d(8)(x)
    plain = F.max_pool2d(x, 2, 2)
    assert blurred.shape == plain.shape
    assert not torch.allclose(blurred, plain, atol=1e-6), (
        "MaxBlurPool output equals plain max-pool — the low-pass filter is not being applied"
    )


# ── (2)/(3) the skip candidates must read the CONFLICT channels, not the recurrent state ────


@pytest.mark.parametrize("name", ["DynamicTopSkip", "FiLMSkip"])
def test_skip_candidates_read_the_dynamic_input_channels(name):
    """`x` is ``[conflict field ; 4 short-term state halves]``. The mechanism is about the FIELD.

    Slicing `x[:, -n_dynamic:]` instead of `x[:, :n_dynamic]` would feed the skip the tail of the
    recurrent state — a plausible-looking arm that tests something else entirely, with correct
    shapes and finite gradients throughout.
    """
    m = _build(name)
    e0s = torch.randn(1, m.conv_base, H, W)
    x = torch.randn(1, m.enc_conv0.in_channels, H, W)

    if name == "FiLMSkip":  # zero-init is the identity; give it non-trivial weights
        torch.manual_seed(3)
        nn.init.normal_(m.film.weight, std=0.1)

    base = m._topskip(e0s, None, x)

    x_field = x.clone()
    x_field[:, : m.n_dynamic] += 5.0  # perturb ONLY the conflict channels
    assert not torch.allclose(base, m._topskip(e0s, None, x_field), atol=1e-6), (
        f"{name}: the top-skip ignored the conflict field — it is reading something else"
    )

    x_state = x.clone()
    x_state[:, m.n_dynamic :] += 5.0  # perturb ONLY the recurrent-state part
    assert torch.allclose(base, m._topskip(e0s, None, x_state), atol=1e-6), (
        f"{name}: the top-skip responded to the recurrent state, which is not the mechanism"
    )


# ── (4) ShallowPool: one pooling, and the receptive field held by dilation ───────────────────


def test_shallow_pool_removes_the_second_downsampling():
    m = _build("ShallowPool")
    assert isinstance(m.pool1, nn.Identity), "pool1 still downsamples — the bottleneck is at 1/4"
    assert m.bottleneck_conv.dilation == (2, 2), (
        "the bottleneck is not dilated — resolution was preserved but the receptive field halved, "
        "which confounds the two things this arm separates"
    )
    assert m.upsample0_head1_reg.kernel_size == (1, 1), (
        "upsample0 still upsamples; with pool1 removed the decoder concat shapes would disagree"
    )


def test_shallow_pool_keeps_the_bottleneck_at_half_resolution():
    """Observable rather than structural: the encoder path must halve once, not twice."""
    m = _build("ShallowPool")
    feat = torch.randn(1, m.conv_base, H, W)
    assert m.pool0(feat).shape[-1] == H // 2
    e1s = torch.randn(1, m.conv_base * 2, H // 2, W // 2)
    assert m.pool1(e1s).shape[-1] == H // 2, "pool1 must be a no-op"


# ── (5) DualStream: the parallel stream must influence the output ────────────────────────────


def test_dual_stream_is_not_inert():
    """A stream that is built but multiplied by zero has finite gradients and changes nothing.

    That is the realistic form of "wired but dead", and the gradient check cannot see it.
    """
    m = _build("DualStream")
    e0s = torch.randn(1, m.conv_base, H, W)
    x = torch.randn(1, m.enc_conv0.in_channels, H, W)
    before = m._topskip(e0s, None, x)
    with torch.no_grad():  # perturb ONLY the parallel stream
        for p in m.stream.parameters():
            p.add_(0.5)
    after = m._topskip(e0s, None, x)
    assert not torch.allclose(before, after, atol=1e-6), (
        "the parallel stream does not affect the top-skip — DualStream is inert"
    )


# ── (6) WideMemory: the state must actually be wider, and the convs must not be ──────────────


def test_wide_memory_widens_only_the_memory():
    from views_hydranet.architectures.wide_memory import MEMORY_WIDTH_FACTOR

    inc = _build("HydraBNUNet06_LSTM4")
    wide = _build("WideMemory")
    assert MEMORY_WIDTH_FACTOR > 1, "a factor of 1 makes this arm a clone of its own control"
    assert wide.base == HID * MEMORY_WIDTH_FACTOR, "the recurrent state was not widened"
    assert wide.conv_base == inc.conv_base, "the conv stack widened too — the arm is confounded"
    lstm = sum(p.numel() for n, p in wide.named_parameters() if n.startswith("W"))
    lstm_inc = sum(p.numel() for n, p in inc.named_parameters() if n.startswith("W"))
    assert lstm > 2 * lstm_inc, f"LSTM barely grew: {lstm_inc} -> {lstm}"


def test_every_candidate_differs_from_the_incumbent_in_output():
    """The blunt backstop: a candidate that produces the incumbent's output tests nothing.

    Weights are seeded identically, so any difference is architectural. `AntiAliasedPool` and the
    zero-init `FiLMSkip` are exempt by construction — the first changes only pooling (checked
    above), the second is deliberately the identity until trained.
    """
    inc = _build("HydraBNUNet06_LSTM4")
    torch.manual_seed(11)
    x = torch.randn(1, IN_CH, H, W)
    with torch.no_grad():
        ref = inc(x, inc.init_hTtime(inc.base, H, W)).reg
    for name in ("DynamicTopSkip", "ShallowPool", "DualStream", "WideMemory"):
        m = _build(name)
        with torch.no_grad():
            out = m(x, m.init_hTtime(m.base, H, W)).reg
        assert not torch.allclose(out, ref, atol=1e-7), (
            f"{name} reproduces the incumbent's output exactly — it would score as a perfect null"
        )
