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
from views_hydranet.architectures.upsample_interp import (  # noqa: E402
    NearestConvUp,
    _UpsampleInterpBase,
)

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


# ── (7) UpsampleConv{1,2}: no transposed conv may survive, on any of the TWELVE paths ────────

#: Every upsample attribute in the incumbent. Replacing only the `reg` paths would leave the gate's
#: placement on the old stage, and EXP-01's finding is that the effect IS the gate's placement.
_UPSAMPLES = tuple(
    f"upsample{stage}_head{head}_{branch}"
    for head in (1, 2, 3)
    for branch in ("reg", "class")
    for stage in (0, 1)
)

#: ⛔ The parameter counts are asserted RELATIONALLY, never against a literal. An earlier version
#: pinned 905,289 / 720,969 here — a tripwire that the first legitimate change to the incumbent
#: would trip, and that would then be "fixed" by editing the constant, after which it guards
#: nothing. What the capacity argument actually claims is a RELATION: the 2x2 remedy is free and
#: the 1x1 remedy is cheaper. That survives any change to the parent.


@pytest.mark.parametrize("name", ["UpsampleConv2", "UpsampleConv1"])
def test_upsample_candidates_replace_all_twelve_transposed_convs(name):
    """One surviving ConvTranspose2d would leave the measured defect on that path."""
    m = _build(name)
    assert len(_UPSAMPLES) == 12, _UPSAMPLES
    for attr in _UPSAMPLES:
        mod = getattr(m, attr)
        assert not isinstance(mod, nn.ConvTranspose2d), f"{attr} is still a ConvTranspose2d"
        assert isinstance(mod, NearestConvUp), f"{attr} is {type(mod).__name__}"
    leftover = [n for n, z in m.named_modules() if isinstance(z, nn.ConvTranspose2d)]
    assert not leftover, f"transposed convs survive at {leftover}"


@pytest.mark.parametrize("name", ["UpsampleConv2", "UpsampleConv1"])
def test_upsample_candidates_hold_the_decoder_shapes(name):
    """The parent concatenates each upsample's output with a skip, so a shape change throws there.

    Checked against the incumbent's own stages rather than against hard-coded numbers.
    """
    inc, m = _build("HydraBNUNet06_LSTM4"), _build(name)
    for attr in _UPSAMPLES:
        in_ch = inc.conv_base * 4 if attr.startswith("upsample0") else inc.conv_base * 2
        res = H // 4 if attr.startswith("upsample0") else H // 2
        feat = torch.randn(1, in_ch, res, res)
        with torch.no_grad():
            assert getattr(m, attr)(feat).shape == getattr(inc, attr)(feat).shape, attr


def test_upsample_conv2_is_exactly_parameter_free():
    """⭐ The capacity confound, asserted as a RELATION rather than against a literal.

    `nearest + Conv2d(3x3)` — the textbook remedy — would be **+33.9%** of the whole model, 5.6x
    the most capacity-confounded arm in the 2026-08-24 dossier (WideMemory, +6.0%), so a win or a
    loss would be uninterpretable. `UpsampleConv2` is the remedy that costs nothing, which is what
    makes it the arm: the `AntiAliasedPool` standard, where a win cannot be capacity.
    """
    n_inc = sum(p.numel() for p in _build("HydraBNUNet06_LSTM4").parameters())
    n = sum(p.numel() for p in _build("UpsampleConv2").parameters())
    assert n == n_inc, f"UpsampleConv2 is no longer parameter-free: {n:,} vs {n_inc:,}"


def test_upsample_conv1_is_strictly_cheaper_than_the_incumbent():
    """A 1x1 kernel is parameter-REDUCING, so a win cannot be capacity (the ShallowPool read)."""
    n_inc = sum(p.numel() for p in _build("HydraBNUNet06_LSTM4").parameters())
    n = sum(p.numel() for p in _build("UpsampleConv1").parameters())
    assert n < n_inc, f"UpsampleConv1 should be cheaper than the incumbent: {n:,} vs {n_inc:,}"


def test_nearest_conv_up_k2_matches_the_transposed_conv_it_replaces():
    """⭐ The LOCAL invariant the whole-model relation rests on, so a failure localises itself.

    A 2x2 kernel over a nearest-upsampled map has exactly the weight count of the
    `ConvTranspose2d(k=2, stride=2)` it replaces — same in/out channels, same 4 taps, same bias.
    Asserted on the modules directly, at widths taken from the incumbent rather than hard-coded.
    """
    inc = _build("HydraBNUNet06_LSTM4")
    for attr in _UPSAMPLES:
        ct = getattr(inc, attr)
        rep = NearestConvUp(ct.in_channels, ct.out_channels, 2)
        n_ct = sum(p.numel() for p in ct.parameters())
        n_rep = sum(p.numel() for p in rep.parameters())
        assert n_rep == n_ct, f"{attr}: replacement {n_rep} != transposed conv {n_ct}"


def test_a_three_by_three_upsample_kernel_is_refused():
    """The disqualified remedy must fail loudly, not be buildable by a later edit."""
    with pytest.raises(ValueError, match="kernel_size must be one of"):
        NearestConvUp(8, 4, 3)


def test_upsample_candidate_without_a_kernel_refuses_to_build():
    """A subclass that forgets `_KERNEL` would silently inherit the incumbent's stages.

    That is the exact false-negative this file exists to prevent: it would train for ~2 GPU-hours
    and report "this architecture does not help" when the architecture was never applied.
    """

    class _Forgot(_UpsampleInterpBase):
        pass

    with pytest.raises(ValueError, match="does not set _KERNEL"):
        _Forgot(IN_CH, HID, 1, 0.0, output_distribution="nb")


@pytest.mark.parametrize("name", ["UpsampleConv2", "UpsampleConv1"])
def test_upsample_candidates_are_not_inert(name):
    """Seeds are identical, so any output difference is architectural."""
    inc, m = _build("HydraBNUNet06_LSTM4"), _build(name)
    torch.manual_seed(11)
    x = torch.randn(1, IN_CH, H, W)
    with torch.no_grad():
        ref = inc(x, inc.init_hTtime(inc.base, H, W)).reg
        out = m(x, m.init_hTtime(m.base, H, W)).reg
    assert not torch.allclose(ref, out, atol=1e-6), f"{name} reproduces the incumbent's output"


def test_upsample_candidate_refuses_a_stage_that_is_not_the_defect():
    """⭐ The fail-loud guard, exercised with a REAL non-conforming module from the real parent.

    The candidate exists to replace a NON-OVERLAPPING transposed conv (`kernel_size == stride`).
    If the parent's decoder ever changes primitive, wrapping whatever is there would produce an arm
    that tests nothing while looking healthy — the false-negative this file exists to prevent. So
    the replacement loop asserts what it is replacing.

    `enc_conv0` is a plain `Conv2d` on the real parent, so pointing `_STAGES` at it is a faithful
    stand-in for "the thing at this name is not what this candidate was written against".
    """
    from views_hydranet.architectures.upsample_interp import UpsampleConv2

    class _PointedAtAConv(UpsampleConv2):
        _STAGES = ("enc_conv0",)

    with pytest.raises(ValueError, match="expects enc_conv0 to be a ConvTranspose2d"):
        _PointedAtAConv(IN_CH, HID, 1, 0.0, output_distribution="nb")


# ── (7b) the NUMERICAL guards. A /falsify guard audit found every test above checks a type, a
#        shape or a parameter count, and NOT ONE checks what the module computes — so the defect
#        could be put back verbatim (zero-stuff + conv ≡ ConvTranspose2d(k=2,s=2)) with all of
#        them green. These assert the arithmetic instead. ─────────────────────────────────────


def _in_degree(mod, c, h=8):
    """Max number of distinct INPUT cells any single OUTPUT cell reads. 1 == no spatial mixing.

    This is the property that separates the remedy from the defect, and it is not a shape, a type
    or a parameter count — which is why nothing above could see it.
    """
    with torch.no_grad():
        base = mod(torch.zeros(1, c, h, h))
        deg = torch.zeros_like(base[0, 0])
        for i in range(h):
            for j in range(h):
                x = torch.zeros(1, c, h, h)
                x[0, :, i, j] = 1.0
                deg += ((mod(x) - base).abs().sum(1)[0] > 1e-8).float()
    return int(deg.max())


def test_upsample_conv2_actually_mixes_neighbours():
    """⭐⭐ THE GUARD THE AUDIT SAID WAS MISSING.

    `ConvTranspose2d(k=2, s=2)` gives every output cell **exactly one** input cell — that IS the
    defect. `UpsampleConv2` must read **more than one**, or it is the defect under a new name.

    Measured: incumbent 1, honest nearest+2x2 max 4. A zero-stuffing `forward` — which is
    mathematically identical to the transposed conv it replaces — scores 1 and passed every other
    test in this file.
    """
    inc = _build("HydraBNUNet06_LSTM4")
    ct = inc.upsample1_head1_reg
    assert _in_degree(ct, ct.in_channels) == 1, "the incumbent should be the non-mixing case"
    rep = NearestConvUp(ct.in_channels, ct.out_channels, 2)
    assert _in_degree(rep, rep.in_ch) > 1, (
        "UpsampleConv2 reads only one input per output — it has become the defect it replaces "
        "(zero-stuffing, or a kernel that cannot straddle a block boundary)"
    )


def test_upsample_conv1_applies_the_SAME_map_at_every_phase():
    """⭐⭐ `UpsampleConv1`'s whole claim is that **no phase pattern can form by construction**.

    It does NOT mix — in-degree 1 is correct for it — so the mixing guard above would wrongly fail
    it. The claim to assert is instead **phase invariance**: all four positions of each 2x2 output
    block must be the identical function of the single input cell they share.

    An added learnable per-phase offset refutes that claim and passed every other test here.
    """
    rep = NearestConvUp(3, 2, 1)
    with torch.no_grad():
        x = torch.randn(1, 3, 4, 4)
        y = rep(x)
    for a in range(2):
        for b in range(2):
            assert torch.allclose(y[:, :, a::2, b::2], y[:, :, 0::2, 0::2], atol=1e-6), (
                f"phase ({a},{b}) differs from (0,0) — UpsampleConv1 acquired a phase "
                "pattern, the one thing it exists to be incapable of"
            )


def test_nearest_upsample_keeps_the_top_left_alignment():
    """A pad flipped to top/left, or a conv applied before the pad, leaves shape and parameter
    count untouched while shifting the field one cell against the skip it is concatenated with.
    Both passed every other test here. Alignment is asserted on a known impulse.
    """
    rep = NearestConvUp(1, 1, 2)
    with torch.no_grad():
        rep.conv.weight.fill_(0.0)
        rep.conv.weight[0, 0, 0, 0] = 1.0  # read only the window's top-left tap
        rep.conv.bias.fill_(0.0)
        x = torch.zeros(1, 1, 4, 4)
        x[0, 0, 0, 0] = 1.0
        y = rep(x)
    assert y[0, 0, 0, 0] == pytest.approx(1.0), (
        "input (0,0) no longer lands on output (0,0) — the upsample/pad alignment is shifted"
    )
    assert y[0, 0, 3, 3] == pytest.approx(0.0), "an impulse at (0,0) reached the far corner"


def test_nearest_conv_up_supports_exactly_the_two_admitted_kernels():
    """Testing only that `3` is refused leaves every other disqualified kernel buildable."""
    assert NearestConvUp._SUPPORTED == (1, 2), NearestConvUp._SUPPORTED
    for k in (0, 3, 4, 5, 7):
        with pytest.raises(ValueError, match="kernel_size must be one of"):
            NearestConvUp(8, 4, k)


def test_a_transposed_conv_with_uneven_overlap_is_also_refused():
    """⭐ The conformance check has TWO branches and only one was ever executed.

    `_PointedAtAConv` above exercises `not isinstance(..., ConvTranspose2d)`. The clause encoding
    the defect's actual definition — `kernel_size == stride`, i.e. non-overlapping — had no test,
    so deleting it left the whole suite green.

    The probe is assembled through the MRO so the parent builds `probe` before the replacement
    loop walks `_STAGES`, which is how a real stage would arrive.
    """
    from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
    from views_hydranet.architectures.upsample_interp import UpsampleConv2

    class _WithUnevenStage(HydraBNUNet06_LSTM4):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            # kernel 4, stride 2: overlapping, i.e. NOT the defect this candidate replaces
            self.probe = nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1)

    class _Probe(UpsampleConv2, _WithUnevenStage):
        _STAGES = ("probe",)

    with pytest.raises(ValueError, match="kernel_size == stride"):
        _Probe(IN_CH, HID, 1, 0.0, output_distribution="nb")


def test_padding_is_applied_before_the_convolution_not_after():
    """Applying the pad AFTER the conv keeps shape and parameter count and survives the alignment
    impulse — but it feeds a hard-zero final row and column into every skip concatenation.

    Asserted directly: a strictly positive input must produce no all-zero border.
    """
    rep = NearestConvUp(2, 2, 2)
    with torch.no_grad():
        rep.conv.weight.fill_(0.25)
        rep.conv.bias.fill_(0.0)
        y = rep(torch.ones(1, 2, 4, 4))
    assert y[..., -1, :].abs().sum() > 0, "the last ROW is all zeros — pad applied after the conv"
    assert y[..., :, -1].abs().sum() > 0, "the last COLUMN is all zeros — pad applied after conv"


def test_nearest_conv_up_refuses_a_channel_mismatch():
    """The channel check exists because the parent CONCATENATES this output with a skip, so a
    mismatch either throws far downstream or mis-pairs the branches. Deleting it left the suite
    green, so nothing was exercising the raise its own docstring justifies.
    """
    rep = NearestConvUp(8, 4, 2)
    with pytest.raises(ValueError, match="built for 8 channels, got 5"):
        rep(torch.randn(1, 5, 4, 4))
