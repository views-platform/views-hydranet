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
artifact — ⛔ but NOT, as an earlier version of this docstring claimed, "in its purest form" with
``kernel == stride`` as "the worst case". **CORRECTED 2026-10-30 against the source.** Odena
attributes the artifact to **uneven overlap**, which arises *"when the kernel size ... is not
divisible by the stride"* — and ``k=2, s=2`` is evenly divisible, i.e. the configuration Odena
RECOMMENDS as a mitigation.

⭐ What the source does say, and it is the claim this candidate rests on: *"while this approach
helps, it is still easy for deconvolution to fall into creating artifacts"*, and **"models with
even overlap often learn kernels that cause similar artifacts."** ⇒ `M92` is an **empirical
instance of that warning in a production model** — the learned per-phase weights diverging — which
is a narrower and better-supported claim than the one it replaces.

⭐⭐ And the source gives this remedy's mechanism in its own words, matching the derivation above:
*"Where deconvolution has a unique entry for each output window, resize-convolution is implicitly
weight-tying."*

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
`reports/2026-10-30_upsample_arm_dossier/PREREG.md` §2.

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

⚠️ NOTED, NOT FIXED (out of scope here): ``anti_aliased_pool.py`` sizes its pools from
``self.base`` (the RECURRENT width) where ``self.conv_base`` (the CONV width) is meant. The two
coincide only while ``state_channels is None`` — true for every config except ``WideMemory``.
These classes avoid the question entirely by reading the widths off the modules they replace.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

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
        # ⭐ `padding=1` for k=2 does the job `nn.ZeroPad2d((0, 1, 0, 1))` used to, INSIDE the
        # convolution, and `forward` crops the surplus leading row/column. Bit-identical output
        # (verified, max|diff| = 0.0) and cheaper than a separate pad op: 22.3 -> 13.3 ms on the
        # /4 -> /2 stage at the production grid.
        #
        # ⛔⛔ `padding_mode="replicate"`, NOT the `"zeros"` default — PREREG Amendment 3. With
        # zero padding the retained output's last row and last column each read one row of hard
        # zeros and the bottom-right corner reads two. MEASURED on a constant field through a
        # uniform kernel: interior 2.0000, last row 1.0000, corner 0.5000 — exactly 2x and 4x.
        # The INCUMBENT has no such artifact at all: `kernel == stride` means no window ever
        # extends past the input, so there is nothing to pad. ⇒ The defect is introduced BY the
        # remedy, which makes it a second difference between arm and control in a programme whose
        # protocol is that `model` is the only key that differs — the same class as Amendment 2's
        # RNG draw (11a) and output scale (11b). Replication costs NO parameter, so `UpsampleConv2`
        # stays at exactly +-0 against the incumbent, which is the whole reason it is the arm.
        #
        # ⚠️ NOTED, NOT DONE: on a 360x720 global raster the left/right edges are a LONGITUDE WRAP,
        # so `"circular"` would be correct in x while `"replicate"` is correct in y. `padding_mode`
        # applies to all four sides uniformly, so splitting them needs explicit per-axis pad ops —
        # a bigger claim than this defect warrants. Recorded, out of scope (PREREG section 12b).
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            padding=1 if kernel_size == 2 else 0,
            padding_mode="replicate" if kernel_size == 2 else "zeros",
        )

    def apply(self, fn: Callable[[nn.Module], None]) -> NearestConvUp:
        """⛔⛔ Do NOT recurse into ``self.conv``. This is the single-variable fix, not a quirk.

        ``utils.init_weights`` dispatches on ``isinstance(m, nn.Conv2d)`` and
        ``nn.ConvTranspose2d`` is **not** a subclass of ``nn.Conv2d``. So the incumbent's twelve
        upsample stages are the only weights in the whole network that the production init pass
        (``training_engine.py:94-95``, ``model.apply(partial(init_weights, config=config))``)
        never touches — they keep torch's default. The moment they become ``nn.Conv2d``, they are
        xavier-initialised.

        Measured at base=32 with the production ``weight_init: 'xavier_norm'``, before this
        override: ``upsample0_head1_reg`` weight std **0.03612** (incumbent) against **0.05131**
        (``UpsampleConv2``) and **0.10234** (``UpsampleConv1``) — i.e. the arm differed from its
        control in INITIALISATION SCHEME as well as mechanism, on 246,336 parameters, 27% of the
        model, and precisely the ones the experiment is about. That is a second uncontrolled
        variable in an experiment whose protocol is that ``model`` is the only key that differs.

        ``Module.apply`` recurses through ``children()`` and is distinct from ``_apply``, which is
        what ``.to(device)`` and dtype casts use — so skipping the subtree here changes
        initialisation only, and device moves are unaffected. ``fn(self)`` is still called, so any
        visitor that legitimately wants this module still sees it.

        ⇒ The conv keeps torch's default init, exactly like the ``ConvTranspose2d`` it replaces.
        """
        fn(self)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_ch:
            err_msg = (
                f"NearestConvUp built for {self.in_ch} channels, got {x.shape[1]} — the parent "
                "concatenates this output with a skip, so a channel mismatch would either throw "
                "downstream or silently mis-pair the branches."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        # ⭐⭐ A 1x1 convolution COMMUTES with nearest-neighbour upsampling — every output cell in
        # a 2x2 block is the same matrix applied to the same replicated value either way — so
        # convolving BEFORE the upsample does identical arithmetic on a quarter of the pixels.
        # Verified bit-identical (max|diff| = 0.0); 8.5 -> 2.7 ms on the /4 -> /2 stage.
        #
        # ⚠️ This matters because the arm's cost lands on the ROLLOUT and on every forward of a
        # delivered global forecast, not on training: at `window_dim: 32` with a 12-step BPTT
        # backward the replacement is 0.97x, but at the 360x720 production grid the twelve stages
        # go 6.37 -> 25.48 GMAC (exactly 4x) and a full forward 589 -> 830 ms.
        if self.kernel_size == 1:
            return self.up(self.conv(x))
        # k=2: `padding=1` adds a row and column on BOTH sides, so drop the leading one to land
        # back on exactly 2H x 2W with the pad effectively on the right and bottom.
        return self.conv(self.up(x))[..., 1:, 1:]


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
        # ⛔⛔ RESTORE THE RNG. Constructing twelve replacement modules draws from the global
        # generator, so every later draw lands at a different position than the incumbent's would.
        # Measured before this: at construction the arm and its same-seed control shared 121
        # parameters with **0** differing — which is why a construction-time check reads clean and
        # two independent review angles stopped there — but after
        # `model.apply(partial(init_weights, ...))`, which runs on every production model,
        # **53 of 121** differed, including `enc_conv0`, `enc_conv1` and `bottleneck_conv`, modules
        # this class does not touch. ⇒ The paired arm-vs-control contrast at n=2 seeds was not a
        # minimal perturbation; it was a full re-draw of the shared conv stack.
        #
        # Capturing and restoring the state makes the replacement RNG-neutral, so the only
        # difference between arm and control is the one the experiment is about.
        _rng_state = torch.random.get_rng_state()
        # ⭐ The widths are READ OFF the module being replaced, never recomputed. Restating the
        # parent's channel plan (`base*4 -> base*2`, `base*2 -> base`) would duplicate it, and the
        # copy would diverge silently the first time the parent's plan changed.
        for name in self._STAGES:
            old = getattr(self, name)
            # Fail loud if the parent's upsample primitive is no longer what this candidate was
            # written against: the whole point is to replace a NON-OVERLAPPING transposed conv, and
            # wrapping something else would produce an arm that tests nothing while looking fine.
            if not isinstance(old, nn.ConvTranspose2d) or old.kernel_size != old.stride:
                err_msg = (
                    f"{type(self).__name__} expects {name} to be a ConvTranspose2d with "
                    f"kernel_size == stride (the defect it removes); found "
                    f"{type(old).__name__} with kernel_size={getattr(old, 'kernel_size', None)} "
                    f"stride={getattr(old, 'stride', None)}. The parent's decoder changed and "
                    "this candidate needs re-deriving, not silently wrapping."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            setattr(self, name, NearestConvUp(old.in_channels, old.out_channels, self._KERNEL))
        torch.random.set_rng_state(_rng_state)


class UpsampleConv2(_UpsampleInterpBase):
    """Candidate 7a: nearest x2 + Conv2d(2x2). **Exactly +-0 parameters** vs the incumbent."""

    _KERNEL = 2


class UpsampleConv1(_UpsampleInterpBase):
    """Candidate 7b: nearest x2 + Conv2d(1x1). **-20.4% parameters**; no phase pattern can form."""

    _KERNEL = 1
