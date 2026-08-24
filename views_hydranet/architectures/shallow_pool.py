"""ShallowPool — one downsampling instead of two (bake-off candidate 4).

The incumbent pools twice, so the bottleneck sits at **1/4** resolution and every placement
decision is made from features reconstructed from a quarter-scale map. This candidate removes
the second pooling: the bottleneck sits at **1/2**, and `bottleneck_conv` is **dilated** so the
receptive field is preserved rather than halved (`Chen2017_DeepLabv3` atrous convolution;
`Dumoulin2016_ConvolutionArithmetic` for the arithmetic).

Spatial bookkeeping — the part that silently breaks if it is wrong:

    e0s  full        pool0    -> e0  1/2
    e1s  1/2         pool1    -> (identity) e1 1/2
    b    1/2
    upsample0 must NOT upsample now, or `cat([up0(b), e1s])` mixes full with 1/2 and throws.
    It becomes a 1x1 conv doing only the channel change (base*4 -> base*2) at constant resolution.
    upsample1 still doubles 1/2 -> full, meeting `e0s` at full. Unchanged.

⚠️ Registered counter-consideration (`01_literature`, `Islam2020_PositionEncoding`): position
information in a CNN comes largely from zero-padding and **accumulates with depth** — deeper layers
encode more. Removing a downsampling stage preserves resolution but shortens the padded-conv chain,
so this arm may *cost* positional encoding even as it preserves detail. It is not unambiguously
good, and that is why it is a separate arm from candidate 5 rather than bundled with it.
"""

from __future__ import annotations

import torch.nn as nn

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4

_UPSAMPLE0 = (
    "upsample0_head1_reg",
    "upsample0_head1_class",
    "upsample0_head2_reg",
    "upsample0_head2_class",
    "upsample0_head3_reg",
    "upsample0_head3_class",
)


class ShallowPool(HydraBNUNet06_LSTM4):
    """Bottleneck at 1/2 resolution, receptive field held by dilation."""

    def __init__(self, *args, dilation: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        base = self.conv_base

        # 1. the second pooling is removed
        self.pool1 = nn.Identity()

        # 2. the bottleneck keeps its receptive field via dilation. padding == dilation keeps the
        #    output size identical for a 3x3 kernel, so nothing downstream changes shape.
        old = self.bottleneck_conv
        self.bottleneck_conv = nn.Conv2d(
            old.in_channels, old.out_channels, 3, padding=dilation, dilation=dilation, bias=False
        )

        # 3. the first decoder upsample must become a pure channel change: b and e1s are now BOTH
        #    at 1/2, so doubling here would make the concat shapes disagree.
        for name in _UPSAMPLE0:
            setattr(self, name, nn.Conv2d(base * 4, base * 2, 1, bias=False))
