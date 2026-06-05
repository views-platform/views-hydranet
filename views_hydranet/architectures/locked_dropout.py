"""Variational (consistent-mask) dropout for autoregressive stability.

ADR-057. Implements the dropout variant of Gal & Ghahramani (2016),
*A Theoretically Grounded Application of Dropout in Recurrent Neural Networks*:
hold the dropout mask fixed across all forward passes of a single sequence
(here, one posterior sample's 36-step autoregressive roll-forward), refreshing
it only between sequences. This removes the per-step white noise that standard
``nn.Dropout`` injects every forward — noise that a free-running recurrence
amplifies into runaway predictions (risk C-113).

The module is a drop-in replacement for ``nn.Dropout``:

* ``eval()`` mode      → identity (no masking), exactly as ``nn.Dropout``.
* ``train()``, ``locked=False`` → standard inverted dropout, fresh mask every
  forward — byte-identical to ``nn.Dropout`` (this is the training path;
  ADR-057 Decision 2a keeps training unchanged).
* ``train()``, ``locked=True``  → the cached mask is reused across forwards
  (keyed by input shape) until ``reset()`` clears it (this is the inference
  path, enabled per posterior sample).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LockedDropout(nn.Module):
    """Inverted dropout with an optionally locked, reusable mask.

    Args:
        p: Dropout probability in [0, 1).

    Attributes:
        locked: When True (and in train mode), the mask is cached per input
            shape and reused across forwards until ``reset()``. When False,
            behaves exactly as ``nn.Dropout`` (fresh mask each forward).
    """

    def __init__(self, p: float):
        super().__init__()
        if not 0.0 <= p < 1.0:
            err_msg = f"LockedDropout: p must be in [0, 1), got {p}"
            raise ValueError(err_msg)
        self.p = p
        self.locked = False
        # Masks are runtime state, keyed by (shape, device, dtype); never persisted.
        self._masks: dict[tuple, torch.Tensor] = {}

    def reset(self) -> None:
        """Clear cached masks. Call once per posterior sample so the next
        sample's trajectory draws a fresh (but internally consistent) mask."""
        self._masks.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identity in eval mode or when p == 0 (matches nn.Dropout semantics).
        if not self.training or self.p == 0.0:
            return x

        # Unlocked: standard inverted dropout, fresh mask every call. This is
        # exactly nn.Dropout — the training path is therefore unchanged.
        if not self.locked:
            return F.dropout(x, p=self.p, training=True)

        # Locked: reuse a cached mask (per input shape) across forwards so the
        # mask is constant over the autoregressive roll-forward.
        key = (tuple(x.shape), x.device, x.dtype)
        mask = self._masks.get(key)
        if mask is None:
            # Inverted dropout: scale kept units by 1/(1-p) so the expectation
            # is preserved (identical convention to nn.Dropout).
            keep = (torch.rand_like(x) >= self.p).to(x.dtype)
            mask = keep / (1.0 - self.p)
            self._masks[key] = mask
        return x * mask

    def extra_repr(self) -> str:
        return f"p={self.p}, locked={self.locked}"
