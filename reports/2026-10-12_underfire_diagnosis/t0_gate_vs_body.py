"""T0 — is the composed zero mass from the GATE or the BODY? (views-hydranet#396)

⚠️ SCOPE (prereg Amendment 1): T0 is weight-specific, so its result applies to "a
production-config HydraNet at global land", NOT to the delivered forecast.

violet_visitor composes with `threshold_gate` at gate_threshold=0.20, so a cell is zeroed
ENTIRELY when g <= tau; otherwise its zero mass is the body's own P_NB(0).
"""
import sys
from pathlib import Path

import numpy as np
import torch

from views_hydranet.distributions.nb_core import NBCore

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import (  # noqa: E402
    GATE_THRESHOLD,
    SB,
    active_cells_top100,
    build_volume,
    grid_rowcol,
    load_vehicle,
    pull,
)

ART = sys.argv[1] if len(sys.argv) > 1 else None
if ART is None:
    raise SystemExit("usage: t0_gate_vs_body.py <path to calibration_model_*.pt>")
SP = Path(__file__).parent
POOL_LO, POOL_HI = 421, 456
WINDOW = list(range(POOL_LO, POOL_HI + 1))          # inside the TRAIN period (<=456)

model, cfg, sha = load_vehicle(ART)
df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
cells = active_cells_top100(df, POOL_LO, POOL_HI)
vol, land, pg_grid = build_volume(df, WINDOW)
r, c = grid_rowcol(cells)
ri, ci = r - 1, c - 1

# Teacher-forced sweep over the window, exactly as training threads `h` (one backward-free pass).
h = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
g_acc, pz_acc = [], []
with torch.no_grad():
    for t in range(vol.shape[1] - 1):
        out = model(vol[:, t], h)
        h = out.h_next
        g = torch.sigmoid(out.cls[0, SB])                      # [H, W] gate probability
        p = out.reg[:, SB * 2:(SB + 1) * 2, :, :]              # [1, 2, H, W] = (mu, theta)
        mu, th = p[0, 0], p[0, 1]
        pz = torch.exp(NBCore.log_prob_zero(mu, th))           # body's own P(Y=0)
        g_acc.append(g[ri, ci].numpy())
        pz_acc.append(pz[ri, ci].numpy())

G = np.stack(g_acc)        # [T-1, 100]
PZ = np.stack(pz_acc)

gate_kills = G <= GATE_THRESHOLD                       # threshold_gate zeroes the WHOLE cell
composed_zero = np.where(gate_kills, 1.0, PZ)

print("=" * 76)
print("T0 — attribution of composed zero mass  (threshold_gate, tau=%.2f)" % GATE_THRESHOLD)
print("=" * 76)
print("  cells %d, steps %d (months %d-%d, inside the TRAIN period)"
      % (len(cells), G.shape[0], WINDOW[0], WINDOW[-2]))
print()
print("  gate probability            median %.4f   p10 %.4f  p90 %.4f"
      % (np.median(G), np.percentile(G, 10), np.percentile(G, 90)))
print("  cell-steps killed by GATE   %.1f%%  (g <= tau)" % (100 * gate_kills.mean()))
if (~gate_kills).any():
    print("  body's own P(Y=0)           median %.4f  (on cell-steps the gate LETS THROUGH)"
          % np.median(PZ[~gate_kills]))
else:
    print("  body's own P(Y=0)           n/a — the gate killed every cell-step")
print("  composed zero mass          median %.4f" % np.median(composed_zero))
print("  composed POSITIVE mass      median %.4f   <- compare 0.4062 delivered / 0.7639 observed"
      % (1 - np.median(composed_zero)))
print()
# P2 said the GATE dominates. Attribute each cell-step to whichever contributes more.
gate_share = gate_kills.astype(float)
body_share = np.where(gate_kills, 0.0, PZ)
print("  mean zero mass contributed by GATE: %.4f" % gate_share.mean())
print("  mean zero mass contributed by BODY: %.4f" % body_share.mean())
dom = "GATE" if gate_share.mean() > body_share.mean() else "BODY"
print("  -> dominant source: %s" % dom)
print("  P2 (the gate dominates): %s" % ("HOLDS" if dom == "GATE" else "FAILS"))
if abs(gate_share.mean() - body_share.mean()) < 0.1 * max(gate_share.mean(), body_share.mean()):
    print("  ⚠️ F5: contributions within 10%% — record as INDISCRIMINATE, not as a finding.")
np.savez(SP / "t0_result.npz", cells=cells, G=G, PZ=PZ, sha=sha)
print("\nwrote t0_result.npz")
