"""D4 — write `mu` over the whole land grid, then run the compression discriminator on the head
that actually has the defect.

WHY.  `2026-10-20_d3_static_channel_ceiling.md` §4 measured the body's per-cell magnitude slope at
b = 1.7301 (sd ratio 0.4146, spearman 0.6870): the body ranks cells well and commits to 41% of the
right spread.  §8 then tried to find the mechanism with a spatial-coarsening discriminator and
could only run it on the GATE, because no saved artifact holds `mu` on the full grid — and the gate
turned out not to have the defect (b 0.39 / 0.94).  So the mechanism is still unidentified.

This script removes that limit.  Same vehicle, same lead-in, same locked-dropout protocol as
`p2_gate_calibration.py`; it collects the sb body params for every land cell at every fed month
instead of the gate alone, and stores the gate too so both heads come from ONE pass and cannot
drift apart.

THE DISCRIMINATOR (unchanged from §8.2, now pointed at the body).  Aggregate prediction and truth
into k x k blocks and recompute b:
  S  spatial over-smoothing  -> b falls toward 1 as k grows (coarsening averages away the very
                                neighbours the decoder averaged in)
  G  global shrinkage        -> b flat in k (weight decay / a timid objective act at every scale)
  R  rank noise              -> b falls toward 1 AND r climbs steeply
Reported in sample (421-456, the window §4's b came from) and out of sample (457-469, the window
p2's gate numbers came from), so neither can be quoted as the other.

COST: measured, not estimated — wall time and peak RSS are printed at the end (⛔ the standing cost
rule).  No training, no pod, CPU.
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from views_hydranet.distributions import get_family
from views_hydranet.distributions.family_loss import FamilyLoss
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
    raise SystemExit("usage: d4_mu_on_the_grid.py <path to calibration_model_*.pt>")
SP = Path(__file__).parent
POSTERIOR_D = 4
#: Optional filename suffix, so a SECOND vehicle can be measured without overwriting the first.
#: ⛔ Default "" keeps every existing filename byte-for-byte: the control's `d4_result.npz` was
#: produced by a checkpoint that NO LONGER EXISTS on disk (see
#: `d4_result_CONTROL_nb_4acbf4d3.PROVENANCE.md`), so an unconditional `out_path` made reading out
#: a second artifact destroy the only copy of the first. Found by a readiness falsification audit.
TAG = ""
for _a in sys.argv[1:]:
    if _a.startswith("--tag="):
        TAG = _a.split("=", 1)[1]

#: ⛔ `--dropout-off` runs ONE deterministic pass with MC dropout disabled, for the C-128 cost test:
#: is the dropout posterior worth 4x the forward passes, against simply drawing more from the head?
#: It lives HERE rather than in a separate script so the two conditions share one forward loop —
#: a cost comparison between two near-identical loops in two files is a comparison of the files.
DROPOUT_OFF = "--dropout-off" in sys.argv[1:]

LEADIN_START, FIRST_KEPT, LAST_FED = 421, 421, 469
IS_LO, IS_HI = 421, 456          # in sample  (the window §4's b = 1.73 came from)
OS_LO, OS_HI = 457, 469          # out of sample (p2's gate window)
EPS = 1e-6

# ⛔ THREAD CAP.  The first launch of this script was uncapped: it took all 12 cores and 2.96 GB and
# made the laptop unusable mid-video-call.  The cap is therefore deliberate and must stay.  8 of 12
# is the "machine is free" setting (2 is the safe one for a run alongside other work).  Note the
# ceiling on what more threads buy: the pass is 4 posterior draws x 49 SEQUENTIAL steps over one
# 360x720 grid, so wall time is dominated by step count, not by intra-op width.
torch.set_num_threads(8)

t_start = time.time()

ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
model, cfg, sha = load_vehicle(ART)
if DROPOUT_OFF:
    # ⛔ `set_locked_dropout(False)` is NOT enough and this is the trap: it clears `locked` but
    # never calls `.eval()`, so a module already switched to train() by an earlier True would keep
    # resampling a FRESH mask every forward — more dropout noise, not less. Put them in eval
    # explicitly, then PROVE it below rather than trusting it.
    from views_hydranet.architectures.locked_dropout import LockedDropout
    n_do = 0
    for m in model.modules():
        if isinstance(m, LockedDropout):
            m.locked = False
            m.eval()
            n_do += 1
    if n_do == 0:
        raise SystemExit("VOID: --dropout-off found no LockedDropout modules to disable.")
    print(f"  --dropout-off: {n_do} LockedDropout modules set to eval()")
else:
    model.set_locked_dropout(True)
for m in model.modules():
    if "BatchNorm" in type(m).__name__ and m.training:
        raise SystemExit(f"VOID: {type(m).__name__} in train() mode — BN would be rewritten.")
npar = FamilyLoss(get_family(cfg["output_distribution"])).n_params
print(f"vehicle sha {sha}  family {cfg['output_distribution']}  n_params {npar}")

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
land_cells = np.sort(df.priogrid_id.unique())
lr, lc = grid_rowcol(land_cells)
lri, lci = lr - 1, lc - 1
months = list(range(LEADIN_START, LAST_FED + 1))
kept = [m for m in months if m >= FIRST_KEPT]
vol, _, _ = build_volume(df, months)
print(f"sweeping {months[0]}-{months[-1]} ({len(months)} steps) x {POSTERIOR_D} passes, "
      f"collecting sb (mu, theta) + gate on all {len(land_cells):,} land cells …")

if DROPOUT_OFF:
    POSTERIOR_D = 1   # one deterministic pass; the ensemble is built from head draws instead

MU = np.zeros((POSTERIOR_D, len(kept), len(land_cells)), dtype=np.float32)
TH = np.zeros_like(MU)
GATE = np.zeros_like(MU)
with torch.no_grad():
    if DROPOUT_OFF:
        # ⛔ PROVE dropout is off before spending the pass: two forwards on the same input must be
        # BIT-IDENTICAL. If any mask is still live they will not be, and a "dropout-off" arm that
        # silently kept dropping would make the whole cost comparison meaningless — the C-324
        # inert-knob signature, pointed the other way.
        h0 = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
        a = model(vol[:, 0], h0).reg
        b = model(vol[:, 0], h0).reg
        if not torch.equal(a, b):
            raise SystemExit(
                "VOID: --dropout-off is not off — two forwards on identical input differ by "
                f"max {float((a - b).abs().max()):.3e}. A LockedDropout is still sampling."
            )
        print("  dropout-off VERIFIED: two forwards on identical input are bit-identical")

    for d in range(POSTERIOR_D):
        model.reset_locked_dropout()
        h = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
        for t, fed in enumerate(months):
            out = model(vol[:, t], h)
            h = out.h_next
            if fed < FIRST_KEPT:
                continue
            i = fed - FIRST_KEPT
            p = out.reg[:, SB * npar:(SB + 1) * npar, :, :]
            MU[d, i] = p[0, 0, lri, lci].numpy()
            TH[d, i] = p[0, 1, lri, lci].numpy()
            GATE[d, i] = torch.sigmoid(out.cls[0, SB])[lri, lci].numpy()
        print(f"  pass {d + 1}/{POSTERIOR_D} done  ({time.time() - t_start:.0f}s)")

mu = MU.mean(0).astype(np.float64)        # [len(kept), n_land] posterior-mean mu
th = TH.mean(0).astype(np.float64)
gt = GATE.mean(0).astype(np.float64)
out_path = SP / f"d4_result{TAG}.npz"
# `family` is stored because the readouts must compose the EMITTED magnitude, and what `mu` means
# is the family's business (Amendment 4). The frozen nb control predates this key; readouts fall
# back to "nb" loudly rather than silently.
np.savez_compressed(out_path, mu=MU, theta=TH, gate=GATE, land_cells=land_cells,
                    months=np.asarray(kept, np.int32), sha=sha,
                    family=cfg["output_distribution"])
print(f"saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

# ---- observed counts on the same cells/months ----
w = df[(df.month_id >= FIRST_KEPT) & (df.month_id <= LAST_FED)]
piv = w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
Y = piv.reindex(land_cells)[kept].to_numpy(np.float64)      # [n_land, len(kept)]
print(f"truth panel {Y.shape}  positive rate {(Y > 0).mean():.6f}")

row, col = lr - 1, lc - 1
mcol = {m: i for i, m in enumerate(kept)}


def window(lo, hi):
    j = [mcol[m] for m in range(lo, hi + 1)]
    return mu[j].mean(0), Y[:, j].mean(1), gt[j].mean(0)


def slope(pred, obs):
    x, y = np.log(pred + EPS), np.log(obs + EPS)
    b, a = np.polyfit(x, y, 1)
    return b, np.corrcoef(x, y)[0, 1], x.std(), y.std(), len(x)


for lab, (lo, hi) in [("IN SAMPLE  421-456", (IS_LO, IS_HI)),
                      ("OUT OF SAMPLE 457-469", (OS_LO, OS_HI))]:
    mu_w, y_w, gt_w = window(lo, hi)
    print("\n" + "=" * 78)
    print(f"1. THE BODY'S SLOPE ON THE FULL GRID — {lab}")
    print("=" * 78)
    for cl, m in [("cells with any event", y_w > 0),
                  ("cells with >= 2 events", (Y[:, [mcol[x] for x in range(lo, hi + 1)]] > 0).sum(1) >= 2),
                  ("all land cells", np.ones(len(land_cells), bool))]:
        b, r, sx, sy, n = slope(mu_w[m], y_w[m])
        print("  %-24s n %6d  b %8.4f  r %+.4f  sd(mu) %.4f  sd(truth) %.4f  ratio %.4f"
              % (cl, n, b, r, sx, sy, sx / sy))
    print("  (100-cell selected set, D3 section 4:  b 1.7301  r +0.7174  sd ratio 0.4146)")

    print("\n" + "=" * 78)
    print(f"2. DISCRIMINATOR — does the BODY's slope relax as the grid is COARSENED?  {lab}")
    print("=" * 78)
    print("   S smoothing => b -> 1.   G global shrinkage => b flat.   R rank noise => b -> 1, r climbs.")
    print("  %4s %8s %10s %10s %10s %10s" % ("k", "blocks", "b", "r", "sd(mu)", "sd(truth)"))
    for k in (1, 2, 4, 8, 16, 32):
        key = (row // k).astype(np.int64) * 10_000 + (col // k)
        uq, inv = np.unique(key, return_inverse=True)
        n = len(uq)
        pm = np.zeros(n); tm = np.zeros(n); cnt = np.zeros(n)
        np.add.at(pm, inv, mu_w); np.add.at(tm, inv, y_w); np.add.at(cnt, inv, 1.0)
        pm, tm = pm / cnt, tm / cnt
        keep = tm > 0
        b, r, sx, sy, nn = slope(pm[keep], tm[keep])
        print("  %4d %8d %10.4f %+10.4f %10.4f %10.4f" % (k, nn, b, r, sx, sy))

    print("\n" + "=" * 78)
    print(f"3. IS THE mu FIELD SMOOTHER THAN THE TRUTH?  {lab}")
    print("=" * 78)
    H, W = 360, 720

    def lag1(v):
        g = np.full((H, W), np.nan)
        g[row, col] = v
        o = []
        for a, b_ in ((g[:, :-1], g[:, 1:]), (g[:-1, :], g[1:, :])):
            kk = np.isfinite(a) & np.isfinite(b_)
            o.append(np.corrcoef(a[kk], b_[kk])[0, 1])
        return o
    for nm, v in [("log mu", np.log(mu_w + EPS)),
                  ("log truth mean", np.log(y_w + EPS)),
                  ("log truth mean, 336-month base rate",
                   np.log(df[df.month_id <= 456].groupby("priogrid_id")[TARGETS[SB]].mean()
                          .reindex(land_cells).to_numpy() + EPS)),
                  ("gate logit (same pass)", np.log(gt_w / (1 - gt_w)))]:
        e, s = lag1(v)
        print("  %-38s east-west %+.4f   north-south %+.4f" % (nm, e, s))
    print("  spearman(mu, truth mean) %+.4f on cells with any event"
          % spearmanr(mu_w[y_w > 0], y_w[y_w > 0]).statistic)

print("\n" + "=" * 78)
print("COST, measured")
print("=" * 78)
print("  wall time %.1f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
