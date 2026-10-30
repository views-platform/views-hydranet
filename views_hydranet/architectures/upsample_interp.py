"""Upsample-interpolation decoders — bake-off candidate 7 (dossier 2026-08-24, EXP-02).

THE DEFECT, MEASURED
--------------------
`M92` / `M93` (`reports/2026-10-30_smoothness_localisation/`) localised the body's spatial
smoothness to the decoder's two upsampling stages. Both are
``ConvTranspose2d(kernel_size=2, stride=2)``, and with **kernel == stride the kernel windows do
not overlap**, so

    out[2i + a, 2j + b] = W[:, :, a, b] @ x[i, j]        for a, b in {0, 1}

Every output cell is a **fixed per-channel scalar multiple of exactly one input cell**, and each of
the four in-block positions gets **its own free weight matrix**. ⇒ The stage **replicates and
rescales; it never interpolates**, and the four phases are free to diverge.

Measured at `upsample1_head1_reg`, dropout off, teacher-forced: lag-1 rho **+0.123146** against
lag-2 **+0.920900**, with channel-wise lag-1 rho spanning **-0.724900 .. +0.718700** — negative
meaning adjacent cells are *anti*-correlated. That is `Odena2016_Deconvolution`'s checkerboard
artifact in its purest form, and ``kernel == stride`` is the worst case because there is no overlap
to average.

End to end: the network is fed a field at rho(1) **+0.344210**, its encoder is roughness-NEUTRAL
(**+0.313470** / **+0.443300**, bracketing its own input), and its output sits at **+0.958400**.
⇒ The whole move belongs to the decoder.

THE TWO REMEDIES, AND WHY THE OBVIOUS ONE IS NOT HERE
-----------------------------------------------------
``nearest + Conv2d(3x3)`` is the textbook fix and it is **DISQUALIFIED on capacity**. With
``base = total_hidden_channels = 32`` the twelve upsample modules hold **246,336** of the
incumbent's **905,289** parameters, and a 3x3 kernel takes them to **553,536** — a **+33.9%**
model, which is **5.6x** the most capacity-confounded arm in the dossier (WideMemory, +6.0%),
whose own table concludes *"no arm is capacity-confounded in the way first feared."* A win or a
loss would be uninterpretable. It is not implemented.

What is implemented are the two remedies that survive that constraint:

  ``UpsampleConv2``  nearest + Conv2d(2x2)   **+-0 parameters, EXACTLY**
      The ``AntiAliasedPool`` standard — the dossier calls a zero-parameter remedy *"the cleanest
      possible experiment in the whole space"*. A 2x2 window over a nearest-upsampled field sees
      four copies of one input cell at the ``(even, even)`` phase and applies the **summed**
      kernel; at the other three phases it **straddles block boundaries and mixes neighbours**. The
      residual phase structure is therefore a **constraint on shared weights**, not a free
      parameter.

  ``UpsampleConv1``  nearest + Conv2d(1x1)   **-20.4% parameters**
      Every phase applies the **same** matrix to the same replicated value, so **no phase pattern
      can exist by construction**. The stage does no spatial mixing; mixing stays with
      ``dec_conv0``/``dec_conv1``, which are already 3x3 and already do it. Being
      parameter-REDUCING, a win cannot be capacity — the same reading the dossier applies to
      ``ShallowPool``'s -16.3%.

⛔ Which one runs on the GPU is decided by ``G-F`` on CPU, not by this docstring. Pre-registration:
`reports/2026-10-30_upsample_arm/PREREG.md` §2.

WHY NEAREST AND NOT BILINEAR
----------------------------
``bilinear`` is itself a smoother, so it would add a second change pointing in the **same direction
as the defect** and confound the arm with the thing it is meant to remove.

SCOPE
-----
All **twelve** upsample modules are replaced — ``upsample{0,1}_head{1,2,3}_{reg,class}``.
Replacing only the ``reg`` path would make the gate's placement incomparable, and EXP-01's finding
is that the effect **is** the gate's placement: *"the body is untouched; this is
entirely the gate's placement degrading through the autoregressive loop."*

``forward`` is untouched, inherited verbatim.

⚠️ NOTED, NOT FIXED (out of scope here): the upsample constructors in the parent size themselves
from ``base`` (the CONV width, stored as ``self.conv_base``), not from ``state`` (the RECURRENT
width, stored as ``self.base``). These classes therefore use ``self.conv_base``.
``anti_aliased_pool.py`` uses ``self.base`` for its pools, which coincides only while
``state_channels is None`` — true for every config except ``WideMemory``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4

logger = logging.getLogger(__name__)


class NearestConvUp(nn.Module):
    """Nearest-neighbour x2 upsample followed by a convolution. Drop-in for ConvTranspose2d(2, 2).

    Output shape is identical to ``ConvTranspose2d(in_ch, out_ch, 2, stride=2)`` — ``[B, out_ch,
    2H, 2W]`` — so the ``torch.cat`` that follows every upsample in the parent is untouched.

    A ``kernel_size=2`` convolution would shrink the doubled map to ``2H-1``, so it is padded by
    one on the bottom and right first. ``kernel_size=1`` needs no padding. Those are the only two
    supported kernels, and 3 is rejected on purpose: it is the textbook remedy and it costs
    +33.9% of the whole model (see the module docstring).
    """

    _SUPPORTED = (1, 2)

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        if kernel_size not in self._SUPPORTED:
            err_msg = (
                f"kernel_size must be one of {self._SUPPORTED}; got {kernel_size}. A 3x3 kernel "
                "here is +33.9% of the model's parameters, 5.6x the most capacity-confounded arm "
                "in the 2026-08-24 bake-off, and is disqualified by PREREG.md section 2."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # (left, right, top, bottom) — pad after upsampling so a 2x2 kernel holds the size.
        self.pad = nn.ZeroPad2d((0, 1, 0, 1)) if kernel_size == 2 else nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_ch:
            err_msg = (
                f"NearestConvUp built for {self.in_ch} channels, got {x.shape[1]} — the parent "
                "concatenates this output with a skip, so a channel mismatch would either throw "
                "downstream or silently mis-pair the branches."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return self.conv(self.pad(self.up(x)))


class _UpsampleInterpBase(HydraBNUNet06_LSTM4):
    """Shared replacement of all twelve ConvTranspose2d upsamples. Not registered directly."""

    #: Set by each concrete subclass. None here so an unset subclass fails loudly.
    _KERNEL: int | None = None

    #: Every upsample attribute in the parent, with the resolution step it performs.
    #: ``upsample0_*``: base*4 -> base*2.  ``upsample1_*``: base*2 -> base.
    _STAGES = tuple(
        f"upsample{stage}_head{head}_{branch}"
        for head in (1, 2, 3)
        for branch in ("reg", "class")
        for stage in (0, 1)
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._KERNEL is None:
            err_msg = (
                f"{type(self).__name__} does not set _KERNEL, so it would inherit the incumbent's "
                "ConvTranspose2d stages and report a null that reads as 'this architecture does "
                "not help' rather than 'this architecture was never applied'."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        b = self.conv_base  # the CONV width — see the module docstring's scope note
        for name in self._STAGES:
            in_ch, out_ch = (b * 4, b * 2) if name.startswith("upsample0") else (b * 2, b)
            setattr(self, name, NearestConvUp(in_ch, out_ch, self._KERNEL))


class UpsampleConv2(_UpsampleInterpBase):
    """Candidate 7a: nearest x2 + Conv2d(2x2). **Exactly +-0 parameters** vs the incumbent."""

    _KERNEL = 2


class UpsampleConv1(_UpsampleInterpBase):
    """Candidate 7b: nearest x2 + Conv2d(1x1). **-20.4% parameters**; no phase pattern can form."""

    _KERNEL = 1
