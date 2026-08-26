"""DynamicTopSkip and FiLMSkip — same information, two primitives (candidates 2 and 3).

The incumbent's only full-resolution path from input to output is a single conv block
(`enc_conv0 -> e0s -> dec_conv1`). The most spatially precise signal available — **where events
actually were last month** — must therefore survive that block, the ConvLSTM mixing, and two 2x2
poolings before it can reach a decision layer. These candidates hand it to the decoders.

**Why two classes and not one.** ADR-061 built exactly this seam for STATIC content, and the v2
CoordConv result retired it: coordinates at the top-skip scored *worse* than encoder-only and were
seed-unstable, and C-230 recorded the lesson as "raw concat is the wrong primitive; use learned
modulation". That verdict was reached with static, position-redundant content. These two arms are
matched on everything except the primitive:

* `DynamicTopSkip` — raw concat, the retired primitive with non-redundant content;
* `FiLMSkip` — learned per-channel modulation, the primitive C-230 recommends.

Run together they separate *primitive* from *content*, which the CoordConv test could not.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4


def _widen_dec_conv1(model: HydraBNUNet06_LSTM4, extra_in: int) -> None:
    """Rebuild every head's `dec_conv1` to accept `extra_in` more input channels.

    There are six of them (3 heads x reg+cls) and they are the layers that consume the top-skip.
    Rebuilding rather than patching keeps weight init identical in kind to the incumbent's.
    """
    for name in (
        "dec_conv1_head1_reg",
        "dec_conv1_head1_class",
        "dec_conv1_head2_reg",
        "dec_conv1_head2_class",
        "dec_conv1_head3_reg",
        "dec_conv1_head3_class",
    ):
        old = getattr(model, name)
        setattr(
            model,
            name,
            nn.Conv2d(
                old.in_channels + extra_in,
                old.out_channels,
                old.kernel_size,
                padding=old.padding,
                bias=old.bias is not None,
            ),
        )


class DynamicTopSkip(HydraBNUNet06_LSTM4):
    """Raw-concat the dynamic input channels onto the full-resolution skip."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _widen_dec_conv1(self, self.n_dynamic)

    def _topskip(self, e0s, coords, x):
        parts = [e0s, x[:, : self.n_dynamic]]
        if coords is not None:
            parts.append(coords)
        return torch.cat(parts, 1)


class FiLMSkip(HydraBNUNet06_LSTM4):
    """Modulate the full-resolution skip by the dynamic input: ``e0s * (1 + gamma) + beta``.

    A 3x3 conv predicts per-channel, per-cell ``(gamma, beta)`` from the dynamic channels.
    **Initialised to zero**, so at step 0 the modulation is exactly the identity and the network
    starts byte-identical to the incumbent — the same default-off discipline the config flags use.
    The skip's channel count is unchanged, so `dec_conv1` is not widened and this arm adds far
    fewer parameters than raw concat.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.film = nn.Conv2d(self.n_dynamic, 2 * self.base, 3, padding=1)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def _topskip(self, e0s, coords, x):
        gamma, beta = self.film(x[:, : self.n_dynamic]).chunk(2, dim=1)
        out = e0s * (1.0 + gamma) + beta
        return torch.cat([out, coords], 1) if coords is not None else out
