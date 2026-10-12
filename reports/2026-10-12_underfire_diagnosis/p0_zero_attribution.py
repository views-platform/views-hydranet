"""P0 — the zero attribution measured on the PRODUCTION posterior, not inferred from one pass.

T0 (views-hydranet#396) attributed the composed zero mass analytically from a SINGLE deterministic
forward pass: `composed_zero = where(g <= tau, 1, P_NB(0))`. Production emits something else:

  * S = D x K = 4 x 4 = 16 columns (`hydranet_inference.py:1578`), not one;
  * each of the D passes runs with **MC-dropout ON** — `model.eval()` plus
    `set_locked_dropout(True)` and one `reset_locked_dropout()` per pass (ADR-057), so every pass
    has its own gate AND its own (mu, theta). `vehicle.load_vehicle` calls `.eval()` only, which
    makes dropout the identity, so T0 measured the posterior MEAN pass, not a posterior draw;
  * the zero mass is a count of REALISED draws, not an analytic P(0).

So T0's attribution is an expectation evaluated at one point, and the emitted object is a 16-column
sample. `t0_verify_on_emitted_draws.py` already found the consequence: the archive emits 0.6250
positive against T0's 0.5450 (opponent's B7, the 0.08 accounting gap).

This script walks that gap in three rungs on ONE axis at a time, so whatever closes it is named:

  A  1 deterministic pass, analytic P(0)      <- T0 as published
  B  D=4 MC-dropout passes, analytic P(0)     <- adds dropout only
  C  D=4 MC-dropout passes, 16 realised draws <- adds sampling; THIS is what production emits

and reports the headline the composition fix rests on: of the emitted zeros, what share is the
BODY's own draw and what share is the GATE's mask. T0 inferred 94% for the body; rung C
measures it.

NOT at issue here: the autoregressive roll-forward. T0 and all three rungs are teacher-forced, so
any residue left after rung C is attributable to free-running and is reported as such, not
absorbed.

⚠️ IN-SAMPLE. Months 421-456 are inside the calibration training partition 121-456 (prereg
Amendment: the leakage caveat). That is accepted: the question is the mechanism of composition,
not skill, and both arms of the attribution are computed on the same cell-steps.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from views_hydranet.distributions.composition import compose_samples
from views_hydranet.distributions.nb_core import NBCore
from views_hydranet.distributions.registry import resolve_family

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
    raise SystemExit("usage: p0_zero_attribution.py <path to calibration_model_*.pt>")
SP = Path(__file__).parent
POOL_LO, POOL_HI = 421, 456
WINDOW = list(range(POOL_LO, POOL_HI + 1))

# The production posterior width, from the delivered config (views-models violet_visitor).
POSTERIOR_D = 4          # n_posterior_samples (MC-dropout passes, model uncertainty)
POSTERIOR_K = 4          # n_head_samples (per-cell family draws, outcome uncertainty)
TORCH_SEED = 42          # the arm's configured torch_seed; the stream differs from the pod run's
                         # (different T and origin), so MC error across passes is reported below.

model, cfg, sha = load_vehicle(ART)
family = resolve_family(cfg["output_distribution"])
if family is None:
    raise SystemExit(f"VOID: no family for output_distribution={cfg['output_distribution']!r}")
df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
cells = active_cells_top100(df, POOL_LO, POOL_HI)
vol, land, pg_grid = build_volume(df, WINDOW)
r, c = grid_rowcol(cells)
ri, ci = r - 1, c - 1
n_steps = vol.shape[1] - 1
npar = family.n_params

# ADR-057: dropout submodules into train() with a LOCKED mask; the rest of the model stays in
# eval() so BatchNorm running stats are untouched. This is the C-184/#289 trap — a bare
# `model.train()` here would rewrite BN buffers across every lead-in step and confound the run.
if not hasattr(model, "set_locked_dropout"):
    raise SystemExit("VOID: the vehicle has no set_locked_dropout; cannot replicate MC-dropout.")
model.set_locked_dropout(True)
for m in model.modules():
    name = type(m).__name__
    if "BatchNorm" in name and m.training:
        raise SystemExit(f"VOID: {name} is in train() mode — BN buffers would be rewritten.")

G = np.zeros((POSTERIOR_D, n_steps, len(cells)), dtype=np.float64)       # gate per pass
PZ = np.zeros_like(G)                                                    # analytic body P(Y=0)
BODY = np.zeros((POSTERIOR_D, n_steps, len(cells), POSTERIOR_K), np.float32)  # log1p body draws

with torch.no_grad():
    for d in range(POSTERIOR_D):
        model.reset_locked_dropout()          # one fresh locked mask per posterior sample
        h = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
        for t in range(n_steps):
            out = model(vol[:, t], h)
            h = out.h_next
            g = torch.sigmoid(out.cls[0, SB])[ri, ci]                    # [100]
            # out.reg is ALREADY activated: the head applies the family's activation
            # (`_family_activation`, HydraBNrecurrentUnet_06_LSTM4.py:119) because the config
            # carries reg_activation=None. Do NOT softplus again.
            p = out.reg[:, SB * npar:(SB + 1) * npar, :, :].permute(0, 2, 3, 1)[0][ri, ci]
            mu, th = p[..., 0], p[..., 1]
            G[d, t] = g.numpy()
            PZ[d, t] = torch.exp(NBCore.log_prob_zero(mu, th)).numpy()
            # Production's per-(pass, step) sub-generator (sampling.py:99) — independent streams
            # so a step's draws depend only on its own params.
            seed_t = (TORCH_SEED + d * 1_000_003 + t * 10_007) & 0x7FFF_FFFF_FFFF_FFFF
            gen_t = torch.Generator().manual_seed(seed_t)
            BODY[d, t] = torch.log1p(family.sample(p, POSTERIOR_K, gen_t)).numpy()

# --- compose exactly as production does: threshold_gate keeps the cell where gate >= tau ---
gate_t = torch.from_numpy(G).float().unsqueeze(-1)                       # [D,T,100,1]
emitted = compose_samples(
    torch.from_numpy(BODY), gate_t.squeeze(-1), "threshold_gate", GATE_THRESHOLD
).numpy()
kept = G >= GATE_THRESHOLD                                               # [D,T,100]

body_zero = BODY == 0.0                                                  # [D,T,100,K]
emit_zero = emitted == 0.0
gate_zero = emit_zero & ~body_zero          # zero ONLY because the mask killed it

print("=" * 78)
print("P0 — zero attribution on the PRODUCTION posterior (S = %d x %d = %d)"
      % (POSTERIOR_D, POSTERIOR_K, POSTERIOR_D * POSTERIOR_K))
print("=" * 78)
print("  cells %d · steps %d (months %d-%d, IN-SAMPLE) · tau %.2f · sha %s…"
      % (len(cells), n_steps, WINDOW[0], WINDOW[-2], GATE_THRESHOLD, sha[:12]))
print()
print("  RUNG A  1 pass, analytic        positive mass (median over cell-steps) %.4f"
      % (1 - np.median(np.where(~kept[0], 1.0, PZ[0]))))
print("  RUNG B  D=4 dropout, analytic   positive mass %.4f"
      % (1 - np.median(np.where(~kept, 1.0, PZ))))
print("  RUNG C  D=4 dropout, 16 draws   positive mass %.4f   <- what production emits"
      % (1 - emit_zero.mean()))
print("          reference: 0.4062 delivered · 0.3750 archive · 0.7639 observed (0.6944 honest)")
print()
print("  ATTRIBUTION of the emitted zeros (rung C, all %d columns):"
      % (POSTERIOR_D * POSTERIOR_K))
print("    emitted zero fraction            %.4f" % emit_zero.mean())
print("    of those, the BODY's own draw    %.1f%%   <- T0 inferred 94%%"
      % (100 * body_zero.sum() / emit_zero.sum()))
print("    of those, the GATE's mask alone  %.1f%%" % (100 * gate_zero.sum() / emit_zero.sum()))
print("    cell-steps killed by the gate    %.1f%%" % (100 * (~kept).mean()))
print()
print("  MC error across the D=%d passes (emitted zero fraction per pass):" % POSTERIOR_D)
per_pass = emit_zero.reshape(POSTERIOR_D, -1).mean(1)
print("    " + "  ".join("%.4f" % v for v in per_pass)
      + "   sd %.4f" % per_pass.std(ddof=1))
print()
print("  WHAT MOVED THE NUMBER (one axis at a time):")
a = 1 - np.median(np.where(~kept[0], 1.0, PZ[0]))
b = 1 - np.median(np.where(~kept, 1.0, PZ))
cc = 1 - emit_zero.mean()
print("    dropout (A->B)                   %+.4f" % (b - a))
print("    realised draws (B->C)            %+.4f" % (cc - b))
print("    residue to the archive's 0.6250  %+.4f  <- attributable to the free-running roll"
      % (0.6250 - cc))
np.savez(SP / "p0_result.npz", cells=cells, G=G, PZ=PZ, BODY=BODY, sha=sha,
         D=POSTERIOR_D, K=POSTERIOR_K, tau=GATE_THRESHOLD)
print("\nwrote p0_result.npz")
