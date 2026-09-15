#!/usr/bin/env python3
"""pf_cost.py — measure the pushforward's TRAINING cost directly, on production shapes.

Replaces `train_time.py`, which scraped tqdm and was wrong. That tool matched any completed
progress bar (`current == total`) and took the last one, so on a real run it picked an EMIT bar out
of the ~250 the posterior sampler emits, not the training bar. Filtering to the training loop's
`month/s` unit narrows it to two bars per run — training and the BN recalibration — with no
reliable way to tell them apart, and on the second smoke the control's bars were SLOWER than the
treatment's. A scrape of a shared, noisy log is the wrong instrument for this.

This times the thing itself: one window at the production window size and sequence length, forward
plus backward, with and without the pushforward term, in one process so the machine state is
shared. Peak allocation is read from torch, not from `nvidia-smi`, so it excludes the CUDA context
and other processes.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from views_hydranet.architectures.registry import get_architecture
from views_hydranet.distributions import get_family
from views_hydranet.distributions.family_loss import FamilyLoss
from views_hydranet.train.training_engine import _SequenceIndices, _process_sequence

FEATS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
CLS = ["by_sb_best", "by_ns_best", "by_os_best"]


class _Sum(nn.Module):
    def forward(self, losses):
        return losses.sum()


def one(weight: float, *, hw: int, T: int, device: torch.device, detach: bool = False):
    """Peak MiB and fwd+bwd seconds for a single window at the given pushforward weight."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    model = (
        get_architecture("HydraBNUNet06_LSTM4")(3, 32, 1, 0.15, output_distribution="nb")
        .float()
        .to(device)
        .train()
    )
    h = model.init_hTtime(model.base, hw, hw).float().to(device)
    torch.manual_seed(1)
    n = len(FEATS) + len(CLS)
    x = (torch.rand(1, T, n, hw, hw, device=device) < 0.03).float() * torch.rand(
        1, T, n, hw, hw, device=device
    ) * 4
    idx = _SequenceIndices(
        FEATS + CLS,
        {
            "features": FEATS,
            "regression_targets": FEATS,
            "classification_targets": CLS,
            "static_channels": [],
        },
    )
    fam = get_family("nb")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = _process_sequence(
        x, model, h,
        criterion_reg=FamilyLoss(fam), criterion_class=nn.BCEWithLogitsLoss(),
        multitaskloss_instance=_Sum(), idx=idx, device=device, family=fam,
        ss_feedback="sample", forecast_composition="soft_gate",
        pushforward_weight=weight, pushforward_detach_state=detach,
    )
    out["total"].backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
    return peak, secs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hw", type=int, default=32, help="window_dim (production: 32)")
    p.add_argument("--steps", type=int, default=336, help="sequence length (calibration: 336)")
    p.add_argument("--repeats", type=int, default=3, help="repeats per condition; the MEDIAN is reported")
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} window={a.hw}x{a.hw} T={a.steps} repeats={a.repeats} (median reported)")

    res = {}
    for label, w, d in (("off  (w=0)", 0.0, False), ("on   (w=0.1)", 0.1, False),
                        ("on   (w=0.1, detached)", 0.1, True)):
        runs = [one(w, hw=a.hw, T=a.steps, device=dev, detach=d) for _ in range(a.repeats)]
        peaks = sorted(r[0] for r in runs)
        times = sorted(r[1] for r in runs)
        res[label] = (peaks[len(peaks) // 2], times[len(times) // 2], times)
        print(f"  {label:24s} peak={res[label][0]:7.0f} MiB  fwd+bwd={res[label][1]:6.2f}s "
              f"(spread {min(times):.2f}-{max(times):.2f})")

    base_p, base_t, _ = res["off  (w=0)"]
    for label in ("on   (w=0.1)", "on   (w=0.1, detached)"):
        pk, tm, _ = res[label]
        print(f"  RATIO {label:24s} time x{tm / base_t:.2f}   peak x{pk / base_p:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
