"""T2 — does #394's non-land dilution actually reach the GRADIENT? (views-hydranet#396)

#394 asserts 75% of magnitude-loss TERMS are non-land, from counting cells. Nobody has shown the
gradient follows: a convolutional head with shared weights might concentrate its gradient on the
live cells regardless. This is the only real test of that issue's mechanism.

Production loss path, `body_supervision='all'` (training_engine.py:481,735):
    apply_body_mask = False  ->  losses_list.append(tw * loss_fn_j(pred_j, target_j))
i.e. `family.nll(params, target)` reduced as a MEAN over the FULL 360x720 grid, ocean included.
So d(loss)/d(params_cell) = (1/N) * d(nll_cell)/d(params_cell) and the per-cell gradient is
directly attributable.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from views_hydranet.distributions import get_family
from views_hydranet.distributions.family_loss import FamilyLoss
from views_hydranet.distributions.nb_core import NBCore

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import SB, build_volume, load_vehicle, pull  # noqa: E402

ART = sys.argv[1] if len(sys.argv) > 1 else None
if ART is None:
    raise SystemExit("usage: t2_gradient_partition.py <path to calibration_model_*.pt>")
SP = Path(__file__).parent
WINDOW = list(range(421, 457))      # inside the TRAIN period

model, cfg, sha = load_vehicle(ART)
df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
vol, land, pg_grid = build_volume(df, WINDOW)
land_t = torch.from_numpy(land)

# Amendment 1 §3: measure the peak, do not trust disk_guard (postmortem §2.5 — it reads the HOST's
# RAM and under-reports true peak 2-3x even corrected).
cuda = torch.cuda.is_available()
if cuda:
    model.cuda()
    vol = vol.cuda()
    land_t = land_t.cuda()
    torch.cuda.reset_peak_memory_stats()

loss_fn = FamilyLoss(get_family(cfg["output_distribution"]))
npar = loss_fn.n_params

# ONE teacher-forced step, then ONE backward — the smallest thing that produces a real gradient.
# ⛔ eval(), NOT train(), and the reason matters. `load_vehicle` already left the model in eval().
# An earlier draft of this script called `model.train()` on the grounds that "the production loss
# path runs in train() mode". That would have broken the measurement two ways:
#   1. dropout_rate is 0.15, so the forward becomes stochastic and the gradient is a single noisy
#      realisation rather than a reproducible quantity;
#   2. BatchNorm would update running_mean/running_var/num_batches_tracked on EVERY step of the
#      no_grad lead-in loop below as well as the graded step. `torch.no_grad()` does NOT prevent
#      that — training_engine._batchnorm_eval's own docstring says so: "it stops gradients, not
#      buffer updates". That is the C-184 / #289 defect class this repo has already been bitten by
#      twice, and it would silently mutate the loaded artifact's buffers before the measurement.
# eval() measures the EXPECTED-VALUE gradient rather than one dropout realisation, which is what a
# partition-share question wants: a single realisation adds noise without moving the shares.
assert not model.training, "model must be in eval() — see the comment above"
h = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:], device=vol.device)
t = len(WINDOW) // 2               # a mid-window step, away from the zero-initialised state
with torch.no_grad():              # thread the state up to t without building a graph
    for s in range(t):
        h = model(vol[:, s], h).h_next

out = model(vol[:, t], h)
params = out.reg[:, SB * npar:(SB + 1) * npar, :, :].permute(0, 2, 3, 1)   # [1,H,W,npar]
params.retain_grad()
target = vol[:, t + 1, SB]                                                # log1p-space, [1,H,W]
loss = loss_fn(params, target)
loss.backward()

peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else float("nan")

# ---- partitions ----
counts = torch.expm1(target)[0]
P_NONLAND = (~land_t).cpu().numpy()
P_ZEROLAND = (land_t & (counts <= 0)).cpu().numpy()
P_ACTLAND = (land_t & (counts > 0)).cpu().numpy()
parts = {"non-land": P_NONLAND, "zero-land": P_ZEROLAND, "active-land": P_ACTLAND}

with torch.no_grad():
    mu, th = params[0, ..., 0], params[0, ..., 1]
    nll_cell = (-NBCore.log_prob(mu, th, counts)).cpu().numpy()            # [H,W]
gnorm_cell = params.grad[0].norm(dim=-1).cpu().numpy()                     # [H,W]

print("=" * 78)
print("T2 — where the magnitude loss and its GRADIENT actually come from")
print("=" * 78)
print("  month %d -> %d, one forward/backward, loss = %.6f"
      % (WINDOW[t], WINDOW[t + 1], loss.item()))
print("  peak CUDA alloc: %.2f GiB  (prereg Amendment 1 §3; postmortem §2.2 prose said ~3 GB)"
      % peak)
print()
hdr = "  %-13s %9s %9s %12s %9s %12s %9s"
print(hdr % ("partition", "cells", "% cells", "loss sum", "% loss", "grad norm", "% grad"))
tot_l, tot_g = nll_cell.sum(), gnorm_cell.sum()
res = {}
for name, m in parts.items():
    n, ls, gs = int(m.sum()), nll_cell[m].sum(), gnorm_cell[m].sum()
    res[name] = (n, ls, gs)
    print(hdr % (name, f"{n:,}", "%.2f%%" % (100 * n / m.size),
                 "%.1f" % ls, "%.2f%%" % (100 * ls / tot_l),
                 "%.4g" % gs, "%.2f%%" % (100 * gs / tot_g)))
print()
nl_term = 100 * res["non-land"][0] / P_NONLAND.size
nl_grad = 100 * res["non-land"][2] / tot_g
print("  #394 asserts non-land is ~75%% of the TERMS -> measured %.2f%%" % nl_term)
print("  the open question was whether the GRADIENT follows -> measured %.2f%%" % nl_grad)
print()
print("  P3 (non-land >= 50%% of gradient norm): %s" % ("HOLDS" if nl_grad >= 50 else "FAILS"))
if nl_grad < 20:
    print("  ⛔ F3 FIRES — #394's mechanism is REFUTED: the dilution is arithmetic without")
    print("     consequence. Say so on #394.")
np.savez(SP / "t2_result.npz", nll=nll_cell, gnorm=gnorm_cell, land=land,
         counts=counts.cpu().numpy(), peak_gib=peak, sha=sha)
print("\nwrote t2_result.npz")
