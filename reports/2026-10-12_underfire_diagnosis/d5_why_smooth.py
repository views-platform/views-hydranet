"""D5 — is the over-smoothing D4 measured ARCHITECTURAL or LEARNED?

Pre-registered in ../2026-10-20_d5_oversmoothing_prereg.md, including Amendment A1. Every reading
and every VOID condition was committed BEFORE this script ran (commits e9bc4b4, a00c975).

D5-A  Same architecture via choose_model(cfg), state dict NOT loaded. Identical 49-step sweep over
      the identical land mask. If the UNTRAINED field is already as smooth as the trained one, the
      architecture imposes the floor and the three training-side candidates are demoted.
        >= 0.85  -> architectural          <= 0.65  -> learned          between -> INCONCLUSIVE
      VOID if sd(log mu) < 0.01 or sd(gate logit) < 0.01, or anything non-finite: an almost-constant
      field makes lag-1 autocorrelation meaningless, and softplus at random init is a real candidate
      for exactly that. A degenerate field is VOID, not smooth.

D5-B  Trained weights, descriptive, NO pre-committed threshold (the prereg says so and says why).
      Run the real lead-in to month 468, then two single steps at 469 -- one clean, one with a single
      land cell's sb input raised by +1 in log1p space -- and measure |delta log mu| against distance.
      Uses the TRAINED weights, so C-325's "measured at initialisation" objection does not apply.

COST: measured and printed, never estimated.
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from views_hydranet.distributions import get_family
from views_hydranet.distributions.family_loss import FamilyLoss
from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import (  # noqa: E402
    SB,
    TARGETS,
    active_cells_top100,
    build_volume,
    grid_rowcol,
    load_vehicle,
    pull,
)

ART = sys.argv[1] if len(sys.argv) > 1 else None
if ART is None:
    raise SystemExit("usage: d5_why_smooth.py <path to calibration_model_*.pt>")
SP = Path(__file__).parent
LEADIN_START, FIRST_KEPT, LAST_FED = 421, 421, 469
IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469
EPS = 1e-6
H, W = 360, 720

# ⛔ THREAD CAP -- see d4_mu_on_the_grid.py. Uncapped, that script took all 12 cores and made the
# laptop unusable mid-video-call. 8 of 12 is the "machine is free" setting.
torch.set_num_threads(8)

t_start = time.time()
ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
model, cfg, sha = load_vehicle(ART)
npar = FamilyLoss(get_family(cfg["output_distribution"])).n_params
print(f"vehicle sha {sha[:16]}…  family {cfg['output_distribution']}  n_params {npar}")

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
land_cells = np.sort(df.priogrid_id.unique())
lr, lc = grid_rowcol(land_cells)
row, col = lr - 1, lc - 1
months = list(range(LEADIN_START, LAST_FED + 1))
kept = [m for m in months if m >= FIRST_KEPT]
vol, _, _ = build_volume(df, months)


def lag1(v):
    """Lag-1 spatial autocorrelation of a land-cell vector, E-W and N-S, NaN off-land."""
    g = np.full((H, W), np.nan)
    g[row, col] = v
    o = []
    for a, b in ((g[:, :-1], g[:, 1:]), (g[:-1, :], g[1:, :])):
        k = np.isfinite(a) & np.isfinite(b)
        o.append(np.corrcoef(a[k], b[k])[0, 1])
    return o


def sweep(m, label):
    """One 49-step pass. Returns per-month (mu, gate) on the land cells."""
    MU = np.zeros((len(kept), len(land_cells)), dtype=np.float64)
    GA = np.zeros_like(MU)
    with torch.no_grad():
        m.reset_locked_dropout()
        h = torch.zeros(1, cfg["total_hidden_channels"], H, W)
        for t, fed in enumerate(months):
            out = m(vol[:, t], h)
            h = out.h_next
            if fed < FIRST_KEPT:
                continue
            i = fed - FIRST_KEPT
            MU[i] = out.reg[0, SB * npar, row, col].numpy()
            GA[i] = torch.sigmoid(out.cls[0, SB])[row, col].numpy()
    print(f"  {label}: swept {len(months)} steps  ({time.time() - t_start:.0f}s)")
    return MU, GA


def eval_mode_or_void(m, who):
    m.set_locked_dropout(True)
    m.eval()
    for mod in m.modules():
        if "BatchNorm" in type(mod).__name__ and mod.training:
            raise SystemExit(f"VOID: {type(mod).__name__} in train() mode on {who}.")


# ======================================================================= D5-A
print("\n" + "=" * 78)
print("D5-A — THE SAME ARCHITECTURE, UNTRAINED")
print("=" * 78)
from views_hydranet.utils.utils import choose_model  # noqa: E402

ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
untrained = choose_model(cfg, torch.device("cpu"))
n_tr = sum(p.numel() for p in model.parameters())
n_un = sum(p.numel() for p in untrained.parameters())
if n_tr != n_un:
    raise SystemExit(f"VOID: architecture mismatch — trained {n_tr} params, untrained {n_un}.")
same = sum(
    1 for (a, b) in zip(model.state_dict().values(), untrained.state_dict().values())
    if a.shape == b.shape and torch.equal(a.float(), b.float())
)
print(f"  identical architecture: {n_un:,} parameters; {same} of "
      f"{len(untrained.state_dict())} state-dict tensors coincide (expected: only constants)")
eval_mode_or_void(untrained, "untrained")
MU_u, GA_u = sweep(untrained, "untrained")

for lab, lo, hi in (("IN SAMPLE  421-456", IS_LO, IS_HI), ("OUT OF SAMPLE 457-469", OS_LO, OS_HI)):
    j = [kept.index(m) for m in range(lo, hi + 1)]
    mu_w, gt_w = MU_u[j].mean(0), GA_u[j].mean(0)
    lmu = np.log(mu_w + EPS)
    lgt = np.log(np.clip(gt_w, EPS, 1 - EPS) / (1 - np.clip(gt_w, EPS, 1 - EPS)))
    print(f"\n  {lab}")
    for nm, v in (("log mu", lmu), ("gate logit", lgt)):
        if not np.all(np.isfinite(v)):
            raise SystemExit(f"VOID (condition 2): non-finite values in untrained {nm}.")
        sd = v.std()
        e, s = lag1(v)
        verdict = ("VOID — degenerate field" if sd < 0.01 else
                   "ARCHITECTURAL" if min(e, s) >= 0.85 else
                   "LEARNED" if max(e, s) <= 0.65 else "INCONCLUSIVE")
        print(f"    untrained {nm:<12} sd {sd:8.4f}   east-west {e:+.4f}   "
              f"north-south {s:+.4f}   -> {verdict}")
    print(f"    mu range  min {mu_w.min():.4g}  median {np.median(mu_w):.4g}  max {mu_w.max():.4g}")
    print(f"    gate range min {gt_w.min():.4g}  median {np.median(gt_w):.4g}  max {gt_w.max():.4g}")

print("\n  for comparison, D4's TRAINED values and the truth reference:")
print("    trained log mu      in-sample +0.8594 / +0.7636   out-of-sample +0.9266 / +0.9227")
print("    trained gate logit  in-sample +0.9598 / +0.9574   out-of-sample +0.9498 / +0.9464")
print("    truth, 336-mo base rate                          +0.6419 / +0.6261")

# ======================================================================= D5-B
print("\n" + "=" * 78)
print("D5-B — HOW FAR DOES ONE INPUT CELL REACH IN ONE STEP?  (trained, descriptive)")
print("=" * 78)
eval_mode_or_void(model, "trained")

top = active_cells_top100(df, IS_LO, IS_HI, n=100)
tr, tc = grid_rowcol(top)
probes, MINSEP = [], 60
for pg, r0, c0 in zip(top, tr - 1, tc - 1):
    if all(max(abs(r0 - r1), abs(c0 - c1)) >= MINSEP for _, r1, c1 in probes):
        probes.append((pg, r0, c0))
    if len(probes) == 6:
        break
print(f"  {len(probes)} probe cells, pairwise Chebyshev separation >= {MINSEP}: "
      + ", ".join(str(p) for p, _, _ in probes))

with torch.no_grad():
    model.reset_locked_dropout()
    h = torch.zeros(1, cfg["total_hidden_channels"], H, W)
    for t, fed in enumerate(months[:-1]):            # lead-in 421..468
        h = model(vol[:, t], h).h_next
    h_fixed = h.clone()
    x = vol[:, -1].clone()                            # the month-469 input
    base = model(x, h_fixed).reg[0, SB * npar].numpy().astype(np.float64)

    prof = {}
    for pg, r0, c0 in probes:
        xp = x.clone()
        xp[0, SB, r0, c0] = xp[0, SB, r0, c0] + 1.0   # +1 in log1p space, one cell, one channel
        pert = model(xp, h_fixed).reg[0, SB * npar].numpy().astype(np.float64)
        d = np.abs(np.log(pert + EPS) - np.log(base + EPS))
        rr, cc = np.ogrid[:H, :W]
        dist = np.maximum(np.abs(rr - r0), np.abs(cc - c0))     # Chebyshev, in cells
        for k in range(0, 41):
            m_ = dist == k
            prof.setdefault(k, []).append(d[m_].max() if m_.any() else np.nan)

print("\n  |delta log mu| against Chebyshev distance from the perturbed cell (max over the ring,")
print("  then MEDIAN over the 6 probes). The perturbation is +1 log1p on ONE cell, ONE channel.")
peak = np.nanmedian(prof[0])
print(f"\n   dist   |d log mu|   as %% of peak      (peak at distance 0 = {peak:.6g})")
reach10 = None
for k in sorted(prof):
    v = np.nanmedian(prof[k])
    pct = 100 * v / peak if peak > 0 else np.nan
    if k <= 12 or k % 4 == 0:
        print(f"   {k:4d}   {v:10.6g}   {pct:9.2f}%")
    if reach10 is None and pct < 10.0:
        reach10 = k
print(f"\n  falls below 10%% of peak at distance {reach10} cells"
      if reach10 is not None else "\n  never falls below 10% of peak within 40 cells")
print("  ⛔ descriptive only — the prereg pre-committed NO threshold for D5-B and forbids")
print("     converting this into a pass/fail after the fact.")

print("\n" + "=" * 78)
print("COST, measured")
print("=" * 78)
print("  wall time %.1f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
