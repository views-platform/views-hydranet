#!/usr/bin/env python
"""single_window_overfit.py — capacity probe: CAN the backbone draw a sharp sparse field at all?

WHY (2026-06-21 proper-score gate, question 3). The over-smoothing might be the conv U-Net +
ConvLSTM backbone (receptive-field averaging + bottleneck diluting sparse signal), not the loss
(Shi's dissent). This isolates that: take the REAL architecture (choose_model, real config) and
overfit it to ONE 32x32 tile with a few hot cells, using a plain per-cell MSE loss (no smoothing
pressure, dropout OFF). Reproduces the sparse field => backbone CAN be sharp, so the blur is the
loss/averaging. Stays blurry while memorizing one frame => the blur is architectural.

Self-contained: real model + real forward, sparse target fed as input channel 0 (tests the
encoder/skip path's ability to preserve sparse high-frequency structure). Writes PNG + npz.

Usage:
    python scripts/single_window_overfit.py [--steps 3000] [--out-dir <dir>]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from views_hydranet.utils.utils import choose_model

H = W = 32
ARCH = {
    "model": "HydraBNUNet06_LSTM4",
    "input_channels": 3,
    "total_hidden_channels": 32,
    "output_channels": 1,
    "dropout_rate": 0.0,  # capacity test: dropout OFF (max capacity, deterministic)
    "output_distribution": "standard",
    # softplus emit (not relu): avoids dead-ReLU collapse in this tiny single-forward setup; it can
    # still concentrate mass on a few cells, which is what the capacity question asks.
    "reg_activation": "softplus",
    "static_channels": [],
}


def make_sparse_target(seed: int = 0) -> np.ndarray:
    """32x32 field: ~7 hot cells (log1p-count magnitudes), zero elsewhere — like real conflict."""
    rng = np.random.default_rng(seed)
    g = np.zeros((H, W), dtype=np.float32)
    vals = [5.0, 4.4, 3.8, 3.0, 2.1, 1.5, 4.0]
    for v in vals:
        r, c = rng.integers(2, H - 2), rng.integers(2, W - 2)
        g[r, c] = v
    return g


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument(
        "--out-dir",
        default="reports/2026-06-21_proper_score_gate_dossier",
    )
    args = ap.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = choose_model(ARCH, device)
    model.train()  # dropout is 0.0 so this is deterministic; train() enables grad path

    target = make_sparse_target()
    y = torch.from_numpy(target)[None, None].to(device)  # [1,1,H,W]
    # Feed the sparse field as input channel 0 (+ zeros) — tests whether the encoder/skip path can
    # preserve sparse high-frequency structure through the bottleneck.
    x = torch.zeros(1, ARCH["input_channels"], H, W, device=device)
    x[:, 0] = y[:, 0]

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for step in range(args.steps):
        opt.zero_grad()
        h = model.init_hTtime(hidden_channels=model.base, H=H, W=W).to(device)
        out = model(x, h)
        # reg head emits one channel per regression target (sb/ns/os) -> shape [1,3,H,W];
        # overfit channel 0 (sb) to the sparse target.
        loss = F.mse_loss(out.reg[:, :1], y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  mse={losses[-1]:.5f}")

    model.eval()
    with torch.no_grad():
        h = model.init_hTtime(hidden_channels=model.base, H=H, W=W).to(device)
        pred = model(x, h).reg[0, 0].cpu().numpy()  # channel 0 = the overfit target

    # readout (no hard verdict; numbers + a picture)
    truth_pos = int((target > 0).sum())
    pred_area = int((pred > 0.5).sum())
    area_ratio = pred_area / max(1, truth_pos)
    # localization: did the hot cells land on the right pixels?
    peak_recovery = float(pred[target > 0].mean() / target[target > 0].mean())
    off_mass = float(pred[target == 0].sum() / max(1e-9, pred.sum()))
    print(
        f"\nfinal mse={losses[-1]:.5f}  truth_pos={truth_pos}  pred>0.5={pred_area}  "
        f"area_ratio={area_ratio:.2f}×  peak_recovery={peak_recovery:.2f}  "
        f"off_target_mass_frac={off_mass:.3f}  pred_max={pred.max():.2f}"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, "overfit_fields.npz"), target=target, pred=pred,
             losses=np.array(losses))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        vmax = float(target.max())
        ax[0].imshow(target, vmin=0, vmax=vmax, cmap="magma")
        ax[0].set_title(f"TARGET (sparse, {truth_pos} hot cells)")
        ax[1].imshow(pred, vmin=0, vmax=vmax, cmap="magma")
        ax[1].set_title(f"OVERFIT PRED (area {area_ratio:.1f}×, peak {peak_recovery:.2f})")
        ax[2].plot(losses)
        ax[2].set_yscale("log")
        ax[2].set_title("MSE vs step")
        for a in ax[:2]:
            a.set_xticks([])
            a.set_yticks([])
        png = os.path.join(args.out_dir, "overfit_capacity.png")
        fig.tight_layout()
        fig.savefig(png, dpi=110)
        print(f"[written] {png}")
    except Exception as e:  # pragma: no cover
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
