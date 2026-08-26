"""DualStream — an HRNet-lite parallel full-resolution stream (bake-off candidate 5).

`Sun2019_HRNet`: "maintains high-resolution representations throughout the network by connecting
parallel multi-resolution convolutions … **rather than recovering high-resolution via
encoder-decoder**", reaching 81.1% mIoU on Cityscapes against DeepLabv3's 78.5% *at lower compute*
(747 vs 1779 GFLOPs).

**This is explicitly HRNet-LITE and must not be described as HRNet.** A faithful HRNet is parallel
multi-resolution branches with repeated cross-scale fusion, and it is not recurrent — grafting the
ConvLSTM into it is a research project of its own. This candidate takes the single idea that
survives compression: a stream that **never downsamples**, running beside the encoder-decoder and
fused into the full-resolution skip before the heads. If it wins, the faithful version becomes the
follow-up rather than the starting point.

⚠️ This arm adds substantial parameters — the capacity confound is registered in `02_design` and the
parameter count must be reported beside any result.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4


class DualStream(HydraBNUNet06_LSTM4):
    """A parallel stream at full resolution, fused into the top-skip."""

    def __init__(self, *args, stream_width: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        base = self.conv_base
        width = base if stream_width is None else stream_width
        # [x ; 4 short-term state halves] — the same input enc_conv0 sees
        in_ch = self.enc_conv0.in_channels

        # Two 3x3 blocks at FULL resolution. No pooling anywhere in this path, which is the point.
        self.stream = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        # Fuse back to `base` channels so `dec_conv1` keeps its input width and the six decoder
        # paths are untouched — the arm varies the FEATURES reaching the heads, not their shape.
        self.fuse = nn.Conv2d(base + width, base, 1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(base)

    def _topskip(self, e0s, coords, x):
        s = self.stream(x)
        out = F.relu(self.bn_fuse(self.fuse(torch.cat([e0s, s], 1))))
        return torch.cat([out, coords], 1) if coords is not None else out
