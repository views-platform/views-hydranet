"""P2 — what `pos_weight` should the gate carry once it owns occurrence alone?

PRE-REGISTERED BEFORE RUNNING. The decision rule below is committed in advance; the measurement
then selects among its branches. Nothing here retrains anything.

THE ALGEBRA THAT MAKES THIS A MEASURABLE QUESTION
-------------------------------------------------
`weighted_bce` with `pos_weight = w` minimises -[w*y*log(s) + (1-y)*log(1-s)]. Setting the
derivative to zero for a cell whose true conditional positive probability is p:

    w*p*(1-s) = (1-p)*s      =>      s* = w*p / (w*p + 1 - p)

So the loss's own optimum is a CALIBRATED gate only at w = 1. At w = 2 it asks for an inflated
gate: at p = 0.69 the optimum is 0.8166, not 0.69.

WHY THAT IS HARMLESS TODAY AND LOAD-BEARING UNDER THE PROPOSED ARM
------------------------------------------------------------------
As shipped (`threshold_gate`, tau = 0.20) the gate is a RANKING device: the cell is kept whole if
`gate >= tau`, so a monotone inflation of the gate changes almost nothing (measured: the threshold
binds on 3.9% of cell-steps, and only 0.52% of all draws are zero because of it).

Under `soft_gate` with a zero-truncated body, `P(body > 0) = 1` by construction
(`truncated_negative_binomial.py:13`, verified: 0 zeros in 256 draws, `prob_positive` degenerate at
1.0). So the emitted occurrence probability IS the gate, exactly:

    emitted P(y > 0) = gate * P(body > 0) = gate

An inflation that was a harmless monotone offset becomes a direct over-forecast of occurrence. This
is the half of the composition that M45 left untouched, and the reason M45's act_ratio compounded.

WHAT IS MEASURED
----------------
The gate's reliability on ALL land cells at the 13 out-of-sample h=1 months (457-469) — the months
P1 established as the only matchable, teacher-forced, weight-matched slice. Per predicted-gate bin:
the observed positive frequency, and the weight the gate's behaviour implies,

    w_implied = s*(1-p) / (p*(1-s))        [inverting s* above]

⛔ Not the 100-cell set. Those were selected ON the outcome (top-100 by observed rate), so a
reliability curve built on them is conditioned on y and would be biased by construction. The full
land field at the same months is the honest sample.

PRE-REGISTERED DECISION RULE (committed before the run)
-------------------------------------------------------
  D1  curve matches s = 2p/(1+p) within tolerance  => the gate achieves its loss optimum
                                                   => set pos_weight = 1.0 for the arm
  D2  curve matches s = p                          => the gate is ALREADY calibrated despite w=2
                                                   => keep pos_weight = 2.0, and record that the
                                                      gate sits below its own loss optimum (the
                                                      same shape as M6 on the body)
  D3  curve matches neither                        => report w_implied and set the arm's pos_weight
                                                      to 2.0 / w_implied, the value that maps the
                                                      achieved weight onto 1.0
  D4  the curve is not monotone, or the gate carries no resolution at all
                                                   => VOID. A miscalibrated-but-ranking gate cannot
                                                      be repaired by a scalar, and the arm needs a
                                                      different design, not a different number.

TOLERANCE, fixed in advance: a branch is "matched" if the median absolute difference between the
observed frequency and that branch's prediction, over bins holding >= 1000 cell-months, is below
half the distance between the two candidate curves at the same p. Otherwise D3.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import (  # noqa: E402
    SB,
    TARGETS,
    build_volume,
    grid_rowcol,
    load_vehicle,
    pull,
)

ART = sys.argv[1] if len(sys.argv) > 1 else None
if ART is None:
    raise SystemExit("usage: p2_gate_calibration.py <path to calibration_model_*.pt>")
SP = Path(__file__).parent
POSTERIOR_D = 4
NP_SEED, TORCH_SEED = 42, 42
BASE_ORIGIN, N_ORIGINS = 456, 13
# P1: the lead-in that matched the archive (residue +0.0034, 95% CI covers zero)
LEADIN_START = 421
LAST_FED = BASE_ORIGIN + N_ORIGINS - 1
SHIPPED_POS_WEIGHT = 2.0            # violet_visitor config_hyperparameters.py:84

ReproducibilityGate.lock_entropy(np_seed=NP_SEED, torch_seed=TORCH_SEED)
model, cfg, sha = load_vehicle(ART)
model.set_locked_dropout(True)
for m in model.modules():
    if "BatchNorm" in type(m).__name__ and m.training:
        raise SystemExit(f"VOID: {type(m).__name__} in train() mode — BN would be rewritten.")

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
land_cells = np.sort(df.priogrid_id.unique())
lr, lc = grid_rowcol(land_cells)
lri, lci = lr - 1, lc - 1
months = list(range(LEADIN_START, LAST_FED + 1))
vol, _, _ = build_volume(df, months)

print(f"sweeping {months[0]}-{months[-1]} x {POSTERIOR_D} passes, "
      f"collecting the gate on all {len(land_cells)} land cells …")
gate = np.zeros((POSTERIOR_D, N_ORIGINS, len(land_cells)), dtype=np.float32)
with torch.no_grad():
    for d in range(POSTERIOR_D):
        model.reset_locked_dropout()
        h = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
        for t, fed in enumerate(months):
            out = model(vol[:, t], h)
            h = out.h_next
            if fed < BASE_ORIGIN:
                continue
            gate[d, fed - BASE_ORIGIN] = torch.sigmoid(out.cls[0, SB])[lri, lci].numpy()

# posterior-mean gate — what the scorer ranks on (`_body_mean_field` docstring: a single pass
# reproduces its AP only to ~11%)
gate_mean = gate.mean(0)                                    # [13, n_land]
truth = np.zeros_like(gate_mean, dtype=bool)
by_month = {m: g for m, g in df.groupby("month_id")}
for n in range(N_ORIGINS):
    g = by_month[BASE_ORIGIN + 1 + n].set_index("priogrid_id").reindex(land_cells)
    truth[n] = (g[TARGETS[SB]].to_numpy() > 0)

s = gate_mean.ravel().astype(np.float64)
y = truth.ravel()
print(f"  {len(s):,} cell-months · observed positive rate {y.mean():.6f}")


def implied_w(sig, p):
    """Invert s* = w p / (w p + 1 - p)  ->  w = s(1-p) / (p(1-s))."""
    sig = np.clip(sig, 1e-9, 1 - 1e-9)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return sig * (1 - p) / (p * (1 - sig))


print("\n" + "=" * 90)
print("P2 — GATE RELIABILITY on all land cells, 13 out-of-sample h=1 months")
print("=" * 90)
print("  weights %s… · posterior-mean gate over D=%d · shipped pos_weight %.1f"
      % (sha[:12], POSTERIOR_D, SHIPPED_POS_WEIGHT))
edges = np.array([0.0, .01, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.01])
print("\n  %-14s %10s %10s %10s %10s %10s %10s"
      % ("gate bin", "n", "mean gate", "observed p", "s*(w=1)=p", "s*(w=2)", "w_implied"))
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (s >= lo) & (s < hi)
    if m.sum() == 0:
        continue
    sg, p = s[m].mean(), y[m].mean()
    w = implied_w(sg, p) if 0 < p < 1 else np.nan
    pred2 = 2 * p / (1 + p)
    rows.append((lo, hi, int(m.sum()), sg, p, pred2, w))
    print("  [%.2f,%.2f)%-5s %10d %10.4f %10.4f %10.4f %10.4f %10s"
          % (lo, hi, "", m.sum(), sg, p, p, pred2,
             "n/a" if not np.isfinite(w) else "%.4f" % w))

big = [r for r in rows if r[2] >= 1000 and 0 < r[4] < 1]
if not big:
    raise SystemExit("VOID (D4): no bin with >=1000 cell-months and a measurable rate.")
obs = np.array([r[4] for r in big])
sg = np.array([r[3] for r in big])
pred1, pred2 = obs, np.array([r[5] for r in big])
err1, err2 = np.median(np.abs(sg - pred1)), np.median(np.abs(sg - pred2))
sep = np.median(np.abs(pred2 - pred1)) / 2
w_med = float(np.median([r[6] for r in big]))

print("\n  BRANCH TEST over %d bins with n>=1000 (tolerance = half-separation %.4f)"
      % (len(big), sep))
print("    median |gate - p|          %.4f   <- D2 (already calibrated, w_eff=1)" % err1)
print("    median |gate - 2p/(1+p)|   %.4f   <- D1 (at its loss optimum, w_eff=2)" % err2)
print("    median w_implied           %.4f" % w_med)
mono = np.all(np.diff(obs) >= -1e-9)
print("    observed frequency monotone in the gate bin: %s" % mono)
print("    resolution: observed p spans %.6f -> %.6f across bins" % (obs.min(), obs.max()))

if not mono or obs.max() / max(obs.min(), 1e-12) < 2:
    verdict, pw = "D4 VOID", None
elif err1 <= sep and err1 < err2:
    verdict, pw = "D2", SHIPPED_POS_WEIGHT
elif err2 <= sep and err2 < err1:
    verdict, pw = "D1", 1.0
else:
    verdict, pw = "D3", SHIPPED_POS_WEIGHT / w_med

print("\n" + "=" * 90)
print("  VERDICT: %s" % verdict)
if pw is None:
    print("  The gate cannot be repaired by a scalar pos_weight — the arm needs a new design.")
else:
    print("  => the arm's loss_class_pos_weight = %.4f   (shipped %.1f)"
          % (pw, SHIPPED_POS_WEIGHT))
print("=" * 90)
np.savez(SP / "p2_result.npz", gate=gate, truth=truth, land_cells=land_cells, sha=sha,
         verdict=verdict, pos_weight=np.nan if pw is None else pw, w_implied=w_med)
print("\nwrote p2_result.npz")
