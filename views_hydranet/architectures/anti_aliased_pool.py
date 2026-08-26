"""AntiAliasedPool — the incumbent with blur-pooled downsampling (bake-off candidate 1).

`Zhang2019_AntiAliasedCNN`: max-pooling **violates the sampling theorem and breaks
shift-equivariance**, and equivariance is lost *progressively at each downsampling layer* — every
layer before the first pool is equivariant, and each subsequent subsampling degrades it further.
Max-pooling decomposes into a dense max followed by naive subsampling; inserting a low-pass filter
between the two restores equivariance while keeping what max-pooling buys.

Why this is the bake-off's cleanest arm: the incumbent has **two** `MaxPool2d(2, 2)`, our entire
measured gap is *placement*, and the remedy adds **zero learnable parameters** — so a win cannot be
attributed to capacity. Zhang reports it also improved accuracy (+0.7-0.9% ImageNet) as a
side-effect of the regularisation, not only consistency (+2.1%).

Only `pool0`/`pool1` are replaced. `forward` is untouched, inherited verbatim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4


class MaxBlurPool2d(nn.Module):
    """Dense max (stride 1) -> fixed binomial blur -> subsample. No learnable parameters.

    The filter is registered as a **buffer**, not a Parameter: it must not be trained, and it must
    The filter is registered as a **buffer**, not a Parameter: it must not be trained, and
    it must travel with `.to(device)` and be in the state dict so a reload blurs the same.
    """

    #: Binomial (Pascal) rows. Bin-5 is Zhang's default and the one carrying the reported numbers.
    _KERNELS = {2: (1.0, 1.0), 3: (1.0, 2.0, 1.0), 5: (1.0, 4.0, 6.0, 4.0, 1.0)}

    def __init__(self, channels: int, filt_size: int = 5, stride: int = 2):
        super().__init__()
        if filt_size not in self._KERNELS:
            raise ValueError(f"filt_size must be one of {sorted(self._KERNELS)}; got {filt_size}")
        self.stride = stride
        self.channels = channels
        row = torch.tensor(self._KERNELS[filt_size], dtype=torch.float32)
        kernel = row[:, None] * row[None, :]
        kernel = kernel / kernel.sum()
        # depthwise: one shared 2-D filter broadcast over channels
        self.register_buffer("blur", kernel[None, None].repeat(channels, 1, 1, 1))
        self.pad = filt_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError(
                f"MaxBlurPool2d built for {self.channels} channels, got {x.shape[1]} — the blur "
                "is depthwise, so a channel mismatch would silently mix or drop channels."
                "depthwise, so a channel-count mismatch would silently mix or drop channels."
            )
        # dense max at stride 1 (the max-pool benefit), then low-pass, then subsample
        x = F.max_pool2d(x, kernel_size=2, stride=1, padding=0)
        x = F.pad(x, (self.pad,) * 4, mode="reflect")
        return F.conv2d(x, self.blur, stride=self.stride, groups=self.channels)


class AntiAliasedPool(HydraBNUNet06_LSTM4):
    """The incumbent with both `MaxPool2d(2, 2)` replaced by `MaxBlurPool2d`."""

    def __init__(self, *args, filt_size: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        # pool0 sees `base` channels (enc_conv0's output); pool1 sees `base * 2` (enc_conv1's).
        self.pool0 = MaxBlurPool2d(self.base, filt_size=filt_size)
        self.pool1 = MaxBlurPool2d(self.base * 2, filt_size=filt_size)
