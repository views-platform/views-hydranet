"""Does `model.train()` in `struct_pen_at_checkpoint.py` overwrite the checkpoint's BN state?

⛔ THE FINDING THIS MEASURES. `struct_pen_at_checkpoint.py:89` calls `model.train()` with one
purpose — to turn MC-dropout ON so the penalty's share can be swept over stochastic draws. But
`train()` is not selective: it puts EVERY module in training mode, including every `nn.BatchNorm2d`
in the vehicle. A BatchNorm in training mode normalises by the CURRENT BATCH's statistics and, as a
side effect, updates its `running_mean`/`running_var` buffers toward them:

    running = (1 - momentum) * running + momentum * batch        # momentum defaults to 0.1

The script then runs **13 forward passes** in that mode (one for the headline table, twelve for the
sweep) before it calls `model.eval()` and measures the row it labels `eval`. If momentum is 0.1,
the weight still carried by the checkpoint's DELIVERED running statistics by then is

    0.9 ** 13 = 0.2542

⇒ roughly three quarters of the `eval` row's normalisation would come from ONE 32x32 window seen
thirteen times, not from the 300 lessons of training that produced the artifact. The row would not
be "the checkpoint with dropout off"; it would be "the checkpoint with dropout off and its
BatchNorm re-estimated on this window".

⚠️ THE SIGNATURE THAT PROMPTED THIS. In the committed control output the headline table's
`SUM struct pen` is **1.860762**, which lies ABOVE the twelve-draw **max 1.837875**. The headline
table is forward #1 — the one pass whose BatchNorm buffers are still almost entirely the
checkpoint's. A first draw lying outside the range of the next twelve is what a drifting buffer
looks like; it is not what dropout noise alone looks like.

⭐ WHAT IS MEASURED, four things, in this order:

  M-A  INVENTORY. How many `_BatchNorm` modules the vehicle has, each one's `momentum` and
       `track_running_stats`, and `num_batches_tracked` before and after. ⛔ If `momentum` is None
       the update is a cumulative average instead, and if `track_running_stats` is False there are
       no buffers to move — either would retire the finding outright, so it is checked FIRST
       rather than assumed.

  M-B  DRIFT. The delivered `running_mean`/`running_var` are snapshotted, the published sequence of
       13 forwards is re-run, and the buffers are compared term by term against the snapshot.

  M-C  WHAT IT COSTS THE PUBLISHED NUMBER. The `eval` row is measured TWICE from the same
       weights: once after the 13 forwards (reproducing the committed row) and once with the
       snapshot restored (what the checkpoint actually delivers). The difference is the published
       row's error.

  M-D  AND WHETHER THE SWEEP ITSELF MOVED. The whole sequence is re-run with dropout ON but every
       BatchNorm held in eval mode, under a re-locked seed so the dropout masks are IDENTICAL.
       Any difference is then attributable to BatchNorm alone. This is the number that says whether
       the twelve-draw sweep — not just the eval row — was reading a moving instrument.

⚠️ THIRTEEN FORWARDS IS A REPLICATION, NOT A DRAW BUDGET. The standing constraint is that
MC-dropout draws stay LOW (1-4). Thirteen is used here because that is exactly what the published
script did, and reproducing its forward count is the whole point. Nothing here estimates anything
from thirteen draws; the draws are the independent variable.

⛔ ITS SWEEP IS NOT COMPARABLE WITH `struct_pen_at_checkpoint.py`'S, AND THE TWO `.out` FILES SIT
SIDE BY SIDE. This script replicates the DEFECTIVE path, so `model.train()` leaves dropout UNLOCKED
in both of its arms — a fresh mask every call. The fixed script locks one mask per draw. Both are
legitimate; they are different estimators (`docs/CICs/LockedDropout.md` §3), and holding BatchNorm
fixed, the difference is worth ~3% on the sweep mean. ⇒ Read M-D's two rows against EACH OTHER,
which is what it is for, and never against the other script's sweep. ⚠️ The `eval` rows ARE
comparable: with dropout off the mask policy cannot matter, and they agree to six decimals.

⛔ ONE WINDOW, ONE SEED, ONE ARTIFACT — the same window, seed and parquet as the script under test,
so every figure is directly comparable to the committed output. This says nothing about any other
window, and a BatchNorm drift measured on a 99.4%-land window is not a bound on a coastal one.

Usage (the control, the artifact the committed output was produced from):

    python reports/2026-10-25_arm_readiness/bn_drift_during_the_sweep.py \\
        /home/simon/views-diagnostics/hf_control_artifact/artifact/calibration_model_20261025_132124.pt
"""

import json
import resource
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm

REPO = Path(__file__).resolve().parents[2]
SP = REPO / "reports/2026-10-12_underfire_diagnosis"
sys.path.insert(0, str(REPO))  # ⛔ FIRST: the working tree must win over installed 0.1.7 (C-347)
sys.path.insert(1, str(SP))

from vehicle import TARGETS, build_volume, load_vehicle, pull  # noqa: E402

from views_hydranet.distributions import get_family  # noqa: E402
from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402
from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate  # noqa: E402
from views_hydranet.train.training_engine import _structure_penalty  # noqa: E402
from views_hydranet.utils.mtloss import MultiTaskLoss  # noqa: E402

#: Every one of these is copied from `struct_pen_at_checkpoint.py` so the two runs are comparable.
PINNED_WEIGHT = 0.416904
WINDOW = 32
POS_WEIGHT = 2.0
DRAWS = 12  # ⇒ 13 forwards in total, matching the script under test

t_start = time.time()
ART = sys.argv[1] if len(sys.argv) > 1 else None
if ART is None:
    raise SystemExit("usage: bn_drift_during_the_sweep.py <path to calibration_model_*.pt>")

cfg = json.loads(Path(f"{ART}.config.json").read_text())
weight = float(cfg.get("struct_penalty_weight") or PINNED_WEIGHT)

device = torch.device("cpu")
ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
model = load_vehicle(ART, device=str(device))
model = model[0] if isinstance(model, tuple) else model

family = get_family(cfg["output_distribution"])
criterion = FamilyLoss(family)
npar = criterion.n_params

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
vol, land, _pg = build_volume(df, list(range(421, 423)))
rows = np.where(land.any(axis=1))[0]
cols = np.where(land.any(axis=0))[0]
r0 = int(rows[len(rows) // 2]) - WINDOW // 2
c0 = int(cols[len(cols) // 2]) - WINDOW // 2
x = vol[:, 0, :, r0 : r0 + WINDOW, c0 : c0 + WINDOW]
y = vol[:, 1, :, r0 : r0 + WINDOW, c0 : c0 + WINDOW]

mtl = MultiTaskLoss(
    is_regression=torch.tensor([1.0] * len(TARGETS) + [0.0] * len(TARGETS)), reduction="sum"
)


def forward():
    return model(x, torch.zeros(1, cfg["total_hidden_channels"], WINDOW, WINDOW))


def measure(model_out):
    """The five scalars the script under test reports, so the rows line up term for term."""
    body, pen = [], []
    for j in range(len(TARGETS)):
        params_j = model_out.reg[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1)
        body.append(criterion(params_j, y[:, j]))
        pen.append(_structure_penalty(family, params_j, model_out.cls[:, j], y[:, j]))
    gate = [
        torch.nn.functional.binary_cross_entropy_with_logits(
            model_out.cls[:, j], (y[:, j] > 0).float(), pos_weight=torch.tensor(POS_WEIGHT)
        )
        for j in range(len(TARGETS))
    ]
    bal = float(mtl(torch.stack(body + gate)))
    bs, ps, gs = float(sum(body)), float(sum(pen)), float(sum(gate))
    return bs, gs, ps, bal, 100.0 * weight * ps / (bal + weight * ps)


# ───────────────────────────── M-A: inventory ─────────────────────────────
bns = [m for m in model.modules() if isinstance(m, _BatchNorm)]
print("=" * 94)
print("M-A  BATCHNORM INVENTORY — checked before anything is assumed about the mechanism")
print("=" * 94)
print(f"  artifact                 {Path(ART).name}")
print(f"  family / dropout         {cfg['output_distribution']} / {cfg.get('dropout_rate')}")
print(f"  _BatchNorm modules       {len(bns)}")
if not bns:
    raise SystemExit("⇒ FINDING RETIRED: the vehicle has no BatchNorm module at all.")
momenta = {m.momentum for m in bns}
tracking = {bool(m.track_running_stats) for m in bns}
print(f"  distinct momentum        {sorted(momenta, key=lambda v: (v is None, v))}")
print(f"  track_running_stats      {tracking}")
print(f"  classes                  {sorted({type(m).__name__ for m in bns})}")
nbt0 = [int(m.num_batches_tracked) for m in bns if m.num_batches_tracked is not None]
print(f"  num_batches_tracked      min {min(nbt0)}  max {max(nbt0)}  (as delivered)")
if tracking == {False}:
    raise SystemExit("⇒ FINDING RETIRED: no layer tracks running statistics, so nothing can move.")
if momenta == {None}:
    print("  ⚠️ momentum is None on every layer ⇒ the update is a CUMULATIVE average, not the")
    print("     0.9**13 geometric decay the finding assumes. The drift below is real but the")
    print("     retained weight is then 1 - 13/(n+13), not 0.2542.")
else:
    m0 = next(iter(momenta))
    print(f"  ⇒ predicted weight still on the DELIVERED buffers after 13 forwards: "
          f"(1-{m0})**13 = {(1 - m0) ** 13:.4f}")

#: ⛔ Snapshot BEFORE any forward pass. `.clone()` because these are live buffers and a reference
#: would be mutated by the very passes this is supposed to measure.
snapshot = [
    (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
    for m in bns
]


def restore():
    for m, (mu, var, nbt) in zip(bns, snapshot, strict=True):
        m.running_mean.copy_(mu)
        m.running_var.copy_(var)
        m.num_batches_tracked.copy_(nbt)


class Drift(NamedTuple):
    """⛔ Named, because `drift_hot[2]` told a reader nothing about what index 2 held."""

    d_mean: np.ndarray
    d_var: np.ndarray
    num_batches_tracked: list[int]


def drift():
    """Per-layer relative movement of the buffers away from the delivered snapshot."""
    dmu, dvar = [], []
    for m, (mu, var, _nbt) in zip(bns, snapshot, strict=True):
        scale = var.sqrt().clamp_min(1e-12)
        dmu.append(float(((m.running_mean - mu).abs() / scale).max()))
        dvar.append(float(((m.running_var - var).abs() / var.clamp_min(1e-12)).max()))
    return np.array(dmu), np.array(dvar)


def sequence(freeze_bn):
    """Re-run the published sequence of 13 forwards, optionally holding BatchNorm in eval mode.

    ⛔ The seed is re-locked and the snapshot restored at entry, so the two calls see IDENTICAL
    dropout masks from IDENTICAL weights and buffers. Any difference between them is BatchNorm.
    """
    restore()
    ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
    model.train()
    if freeze_bn:
        for m in bns:
            m.eval()
    first = measure(forward())
    sweep = np.array([measure(forward()) for _ in range(DRAWS)])
    # ⛔ Read the counter HERE, not after the function returns. My first version printed it at
    # report time, by which point the second sequence had restored the snapshot — so it showed
    # 10410 -> 10410 and said the counter never moved, which is the opposite of what happened.
    after = Drift(*drift(), [int(m.num_batches_tracked) for m in bns])
    model.eval()
    ev = measure(forward())
    return first, sweep, ev, after


# ───────── M-B/M-C: the published sequence, reproduced ─────────
first_hot, sweep_hot, eval_hot, drift_hot = sequence(freeze_bn=False)

#: The eval row the checkpoint actually delivers: same weights, pristine buffers, dropout off.
restore()
model.eval()
eval_pristine = measure(forward())

# ───────── M-D: the same sequence with BatchNorm held still ─────────
first_cold, sweep_cold, eval_cold, drift_cold = sequence(freeze_bn=True)

HDR = "  %-22s %10s %10s %10s %10s %8s"
ROW = "  %-22s %10.6f %10.6f %10.6f %10.6f %7.1f%%"

print()
print("=" * 94)
print("M-B  HOW FAR THE DELIVERED BUFFERS MOVED over the 13 published forwards")
print("=" * 94)
for name, d in (("|Δrunning_mean| / sd", drift_hot.d_mean),
                ("|Δrunning_var| / var", drift_hot.d_var)):
    print(f"  {name:<24s} max {d.max():.4f}   mean {d.mean():.4f}   "
          f"layers over 10%: {int((d > 0.10).sum())}/{len(d)}")
nbt1 = drift_hot.num_batches_tracked
print(f"  num_batches_tracked       {min(nbt0)}..{max(nbt0)}  ⇒  {min(nbt1)}..{max(nbt1)}")
print(f"  and with BatchNorm frozen, for contrast: "
      f"max |Δmean|/sd {drift_cold.d_mean.max():.2e}, max |Δvar|/var {drift_cold.d_var.max():.2e}")

print()
print("=" * 94)
print("M-C  WHAT IT COST THE PUBLISHED `eval` ROW")
print("=" * 94)
print(HDR % ("row", "body", "gate", "pen", "balancer", "share%"))
print(ROW % ("eval, buffers moved", *eval_hot))
print(ROW % ("eval, buffers pristine", *eval_pristine))
delta = np.array(eval_hot) - np.array(eval_pristine)
print("  %-22s %10.6f %10.6f %10.6f %10.6f %+7.1fpp" % ("Δ (published − true)", *delta))
print(f"  ⇒ the committed row reports a penalty of {eval_hot[2]:.6f} where the checkpoint as "
      f"delivered gives {eval_pristine[2]:.6f}")

print()
print("=" * 94)
print("M-D  AND WHETHER THE TWELVE-DRAW SWEEP ITSELF WAS READING A MOVING INSTRUMENT")
print("=" * 94)
for lab, first, sweep in (("train() as published", first_hot, sweep_hot),
                          ("BatchNorm frozen", first_cold, sweep_cold)):
    print(f"  {lab}:")
    print(HDR % ("stat", "body", "gate", "pen", "balancer", "share%"))
    print(ROW % ("forward #1 (headline)", *first))
    for name, v in (("sweep mean", sweep.mean(0)), ("sweep sd", sweep.std(0, ddof=1)),
                    ("sweep min", sweep.min(0)), ("sweep max", sweep.max(0))):
        print(ROW % (name, *v))
    inside = sweep[:, 2].min() <= first[2] <= sweep[:, 2].max()
    print(f"    forward #1's penalty {first[2]:.6f} lies "
          f"{'INSIDE' if inside else '⛔ OUTSIDE'} the sweep range "
          f"[{sweep[:, 2].min():.6f}, {sweep[:, 2].max():.6f}]")
    print()

print("=" * 94)
print("COST, measured")
print("=" * 94)
print("  wall time %.2f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
