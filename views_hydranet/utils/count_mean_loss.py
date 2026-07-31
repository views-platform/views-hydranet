"""count_mean_loss.py — MSE in COUNT space (port of views-lstm-lab count_mean_mse, EXP-07).

The log1p body losses (mse/shrinkage) fit the count-space MEDIAN; for heavy-tailed conflict counts
the mean >> median, so they systematically UNDER-predict magnitude (audit: pred_max ~70 vs truth
~2500). count_mean MSEs in COUNT space, whose minimizer is E[y|x] (the arithmetic mean), directly
targeting that magnitude. In the minimal lab this broke the in-sample ceiling: mcr_pos 0.10->0.62.

⚠️ KNOWN OUT-OF-SAMPLE FAILURE (lab ADR-002, EXP-11): that 0.62 is IN-SAMPLE; on the validation
regime shift mcr_pos collapsed to ~0.14 (NOT ADOPTED in the lab). Targeting the count mean overfits
the magnitude to the calibration period. Treat any in-sample magnitude lift as suspect until judged
on the validation partition (FAO-02). This is a transfer-test lever, not a fix.

STABILITY (matches the lab's tradeoff_trainer recipe — the bare loss explodes without it):
  1. expm1 the log1p body output to counts, clamped to ~10x data_max (the TRAINING clamp,
     log_ceil ~= log1p(480k) for calibration; far tighter than a float-overflow guard), floor 0.
  2. divide the count-space MSE by the squared positive scale mean(y_pos^2) (per batch) so the
     gradient is bounded — without it, targets up to ~48k give ~1e9 squared errors and exploding
     gradients (the original port's 1e16->1e37 runaway).
  3. (grad-clip is applied by the training engine, not here.)

Intended pairing: output_distribution='standard' (E[y]=expm1(body)) and NO hurdle mask (train on
all cells, so the minimizer is the UNCONDITIONAL count mean). needs_latent=False.
"""

from __future__ import annotations

import torch
from torch import nn

# log1p(10 x calibration data_max ~= 48_183) ~= 13.1 — the lab's count-space cap (10x data_max),
# used here as a TRAINING clamp on predicted counts so expm1 cannot run away. NOT a float-overflow
# guard (that would be ~30); this is a much tighter magnitude bound, the actual stabilizer.
_TRAIN_LOG_CEIL = 13.1
_NORM_FLOOR = 1.0  # floor for the squared-scale normalizer (avoids div-by-zero on all-zero)


class CountMeanMSELoss(nn.Module):
    """Squared error in count space, normalized by the squared positive scale; minimizer E[y|x]."""

    needs_latent = False

    def __init__(self, log_ceil: float = _TRAIN_LOG_CEIL, norm_floor: float = _NORM_FLOOR) -> None:
        super().__init__()
        self.log_ceil = log_ceil
        self.norm_floor = norm_floor

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_c = torch.expm1(pred.clamp(max=self.log_ceil).to(torch.float64)).clamp(min=0.0)
        target_c = torch.expm1(target.to(torch.float64))
        pos = target_c > 0
        norm = (target_c[pos] ** 2).mean() if pos.any() else target_c.new_tensor(self.norm_floor)
        norm = norm.clamp(min=self.norm_floor)
        return (((pred_c - target_c) ** 2).mean() / norm).to(pred.dtype)
