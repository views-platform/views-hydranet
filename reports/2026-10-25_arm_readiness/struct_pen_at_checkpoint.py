"""Measure the structure penalty's strength AT A TRAINED CHECKPOINT, not at initialisation.

⛔ THE FAILURE THIS REPLACES. `2026-10-21_structure_penalty_arm_prereg.md` §3b and §3c both close
the derived-weight discussion with *"the run's per-task trajectory log is what would show the two
terms drifting apart"*. **It would not.** `struct_pen` is folded into `loss` at
`training_engine.py:1128` and its magnitude is recorded nowhere — not in `_TRAJECTORY_BASE_COLUMNS`
(six names, none a penalty), not in the per-task columns (body/gate only), not in the step's return
dict (`reg`/`cls` are built from the balancer's 6-vector, which the penalties are added *outside*
of), and not in any of the file's three `wandb.log` calls. Registered as **C-365**; the sentence
itself is **C-303** occurrence #19. Found by `/falsify "you are ready for a pod"` on 2026-10-25.

⛔ AND WHY THE OBVIOUS FIX DOES NOT HELP THIS ARM. The remedy C-365 asks for is a trajectory column
per additive term. That is a change to `views_hydranet`, and `tools/podrun/pod_run_model.sh:172`
installs `views-hydranet~=0.1.1` **from PyPI** — the pod runs the published **0.1.7** wheel, whose
`training_engine.py` is byte-identical to this working tree. ⇒ A column added here would not exist
on the arm run without publishing 0.1.8. The column is still the right fix for the *next* arm; it
cannot be the fix for *this* one.

⭐ WHAT THIS DOES INSTEAD, and why it answers the real question. The penalty is a deterministic
function of the model's parameters and the truth, so it can be recomputed from a delivered
checkpoint with no training and no pod — exactly as `struct_weight_derivation_v3.py` does at
initialisation, but with the TRAINED weights loaded. Run it on the control now and on the arm's
delivered artifact after, and the pair answers *"did the term stay at the strength it was tuned
to?"* at the two points that actually bear on the verdict.

⚠️ WHAT IT STILL DOES NOT GIVE, stated because the sentence it replaces overclaimed in exactly this
way: this is **two points, not a trajectory**. It cannot show *when* a term drifted, or a spike
that resolved, or a divergence that reversed. C-325's warning is narrowed, not retired — and if the
init-to-final move is large, that is a reason to publish the column before the next arm, not a
reason to trust an interpolation between two points.

⚠️ ONE WINDOW, ONE SEED, chosen by LAND COVERAGE only — never by activity (C-351's population slot).
The same window, seed and parquet as `struct_weight_derivation_v3.py`, so every figure here is
directly comparable to §3c's table.

⛔⛔ AMENDMENT 1, 2026-10-27 — THE FIRST RUN OF THIS SCRIPT OVERWROTE THE CHECKPOINT'S BATCHNORM
STATE, AND EVERY ABSOLUTE NUMBER IN `struct_pen_at_checkpoint_CONTROL.out` WAS LOW. The line that
enabled MC-dropout was a bare `model.train()`. That is not selective: all 15 `nn.BatchNorm2d`
layers went into training mode too, normalising by the current batch and updating their running
buffers toward it at momentum 0.1. The 13 forwards below therefore left only 0.9**13 = 0.2542 of
the delivered statistics in place before the row labelled `eval` was measured.

Measured directly, by `bn_drift_during_the_sweep.py` on this same artifact, window and seed:

  · all 15 layers moved; |Δrunning_mean| up to 3.49 sd (mean 0.43), |Δrunning_var| up to 2.61x
    its own value (mean 0.81); `num_batches_tracked` 10410 → 10423, exactly +13
  · the `eval` row: penalty 2.222250 measured against 2.589139 with the buffers intact (−14.2%);
    body −12.8%, gate −16.1%, balancer −15.3%
  · the twelve-draw sweep, by more: penalty mean 1.756 against 2.532 (−31%)
  · with BatchNorm held in eval mode the same 13 forwards move the buffers 0.00e+00

⭐ WHAT IT DID NOT CHANGE, and the reason the verdict stands: the SHARE. 87.6% → 87.4% on the
`eval` row, 85.4% → 86.8% on the sweep. The penalty and the balancer fall together, so the ratio
nearly cancels, and §3c's conclusion — 11.3% of the loss at initialisation, ~87% at the checkpoint
— is a 7.7x move that a 20% error in the absolute terms does not reach. ⚠️ Both ends of that
comparison are sound for a second reason: `struct_weight_derivation_v3.py` has the same bare
`model.train()`, but it builds FRESH weights with `choose_model` and does ONE forward, so it has no
delivered buffers to corrupt.

⛔ THE FIX IS NOT NEW KNOWLEDGE, WHICH IS THE WORST PART. `p0_zero_attribution.py`,
`p1_matched_h1.py` and `t2_gradient_partition.py` — all three older, all three in this same
programme — use `set_locked_dropout` instead and say why in comments; one of them asserts that no
BatchNorm is in train mode and calls it "the C-184/#289 trap, which my own T2 nearly walked into".
This was a regression from documented practice, not a gap in it.

Usage, with the control as the baseline:

    python reports/2026-10-25_arm_readiness/struct_pen_at_checkpoint.py \\
        /home/simon/views-diagnostics/hf_control_artifact/artifact/calibration_model_20261025_132124.pt
"""

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
SP = REPO / "reports/2026-10-12_underfire_diagnosis"
sys.path.insert(0, str(REPO))  # ⛔ FIRST: the working tree must win over installed 0.1.5 (C-347)
sys.path.insert(1, str(SP))

from vehicle import TARGETS, build_volume, load_vehicle, pull  # noqa: E402

from views_hydranet.distributions import get_family  # noqa: E402
from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402
from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate  # noqa: E402
from views_hydranet.train.training_engine import _STRUCT_BLOCKS, _structure_penalty  # noqa: E402
from views_hydranet.utils.mtloss import MultiTaskLoss  # noqa: E402

#: §3c's pinned weight. ⛔ Read from the config sidecar when the artifact records one, so an ARM
#: artifact reports the weight it actually trained under rather than this default.
PINNED_WEIGHT = 0.416904
WINDOW = 32
POS_WEIGHT = 2.0  # the roster's loss_class_pos_weight on heavy_freighter

t_start = time.time()
ART = sys.argv[1] if len(sys.argv) > 1 else None
if ART is None:
    raise SystemExit(
        "usage: struct_pen_at_checkpoint.py <path to calibration_model_*.pt>\n"
        "Run it on the CONTROL first, so the arm has a trained baseline to be compared against."
    )

cfg = json.loads(Path(f"{ART}.config.json").read_text())
weight = float(cfg.get("struct_penalty_weight") or PINNED_WEIGHT)
declared = "struct_penalty_weight" in cfg

ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
device = torch.device("cpu")

# ⛔ TRAINED weights. This is the one line that distinguishes this script from the derivation, and
# it is the whole point: `load_vehicle` also runs the F6 identity gate on the sidecar.
model = load_vehicle(ART, device=str(device))
model = model[0] if isinstance(model, tuple) else model

# ⛔ AMENDMENT 1, 2026-10-27. This line was `model.train()`, and that was wrong — see the
# amendment block in the module docstring for the measurement. `train()` is not selective: it put
# all 15 BatchNorm2d layers into training mode too, so each of the 13 forwards below overwrote a
# tenth of the checkpoint's delivered running statistics. `set_locked_dropout` turns MC-dropout on
# and nothing else, which is what this script wanted and what three older scripts in this same
# programme already do (`p0_zero_attribution.py`, `p1_matched_h1.py`, `t2_gradient_partition.py`).
model.set_locked_dropout(True)
for _m in model.modules():
    if isinstance(_m, torch.nn.modules.batchnorm._BatchNorm) and _m.training:
        raise SystemExit(
            f"VOID: {type(_m).__name__} is in train() mode — its running statistics would be "
            "rewritten by the sweep below, and the `eval` row would report a checkpoint "
            "re-normalised on this one window. Asserted, not assumed."
        )

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

out = model(x, torch.zeros(1, cfg["total_hidden_channels"], WINDOW, WINDOW))

# ⛔ C-364: a slice past the head's width returns an EMPTY tensor in torch rather than raising, so
# a three-name loop over a two-target head would build a plausible table. Assert the width.
expected = len(TARGETS) * npar
if out.reg.shape[1] != expected:
    raise SystemExit(
        f"VOID (C-364): out.reg has {out.reg.shape[1]} channels, expected "
        f"{expected} = {len(TARGETS)} targets x {npar} {cfg['output_distribution']} params. "
        "A positional slice would have returned empty tensors and produced a plausible table."
    )

mtl = MultiTaskLoss(
    is_regression=torch.tensor([1.0] * len(TARGETS) + [0.0] * len(TARGETS)), reduction="sum"
)


def measure(model_out):
    """The four terms the loss is built from, for one forward pass."""
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
    return bs, gs, ps, bal, 100.0 * weight * ps / (bal + weight * ps), body, pen


def forward():
    return model(x, torch.zeros(1, cfg["total_hidden_channels"], WINDOW, WINDOW))


b_sum, g_sum, p_sum, balanced, share, body, pen = measure(out)
w_pen = weight * p_sum

# ⛔ SWEEP THE FREE PARAMETER BEFORE REPORTING THE SCALAR. With MC-dropout on, every forward is a
# different draw and the share above is ONE sample of a random variable.
# ⚠️ An earlier hand-run of this sweep reported sd EXACTLY 0.000000 over 12 draws because dropout
# was off and it swept nothing. An exact zero across a stochastic sweep is the tell.
# ⛔ AND AMENDMENT 1 CREATED A SECOND WAY TO GET THAT ZERO. `set_locked_dropout` LOCKS the mask
# across forwards — that is its purpose, one coherent mask per posterior sample — so without
# `reset_locked_dropout()` all twelve draws would share one mask and sd would be 0.000000 again,
# for a new reason. The guard is unchanged: a zero sd here voids the run.
DRAWS = 12


def draw():
    model.reset_locked_dropout()  # ⛔ one fresh mask per draw; see above
    return measure(forward())[:5]


sweep = np.array([draw() for _ in range(DRAWS)])
# ⛔ AMENDMENT 1a, same day, caught by cross-checking this row against the same quantity measured
# independently in `bn_drift_during_the_sweep.py` (2.589 there against 2.620 here). My first fix
# wrote `model.set_locked_dropout(False)` here on the reasoning that `model.eval()` would unfreeze
# BatchNorm. BOTH halves were wrong. `set_locked_dropout(False)` only clears the `locked` FLAG —
# the submodules stay in train mode, so dropout keeps firing and resamples on every call
# (`tests/test_locked_dropout.py:123`: "locked False == per-step resampling (the bug)"). The row
# labelled `eval` was a thirteenth dropout draw. And `model.eval()` has nothing to unfreeze: with
# amendment 1 applied, BatchNorm never left eval mode. ⇒ `model.eval()` was right all along; only
# the `train()` at the top was ever the defect.
model.eval()
eval_row = measure(forward())[:5]
#: ⛔ The label must not be able to lie again: with dropout genuinely off the forward is
#: deterministic, so two of them agree bit for bit. If they do not, dropout is still firing.
if measure(forward())[:5] != eval_row:
    raise SystemExit(
        "VOID: two consecutive `eval` forwards disagree, so dropout is still active and the row "
        "labelled `eval` is another stochastic draw. This is amendment 1a's defect returning."
    )
if not np.isfinite(sweep[:, 2]).all() or sweep[:, 2].std(ddof=1) == 0.0:
    raise SystemExit("VOID: the penalty's sd across the sweep is exactly 0 — nothing was swept.")

print("=" * 94)
print("THE STRUCTURE PENALTY AT A TRAINED CHECKPOINT")
print("=" * 94)
print(f"  views_hydranet       {Path(__import__('views_hydranet').__file__)}")
print(f"  artifact             {Path(ART).name}")
print(f"  family / dropout     {cfg['output_distribution']} / {cfg.get('dropout_rate')}")
print(f"  struct_penalty_weight{'':1s} {weight:.6g}"
      f"   ({'from the sidecar' if declared else 'NOT in the sidecar — §3c default'})")
print(f"  block sizes          {_STRUCT_BLOCKS}")
print(f"  window               rows {r0}-{r0 + WINDOW}, cols {c0}-{c0 + WINDOW}, "
      f"land {100 * land[r0:r0 + WINDOW, c0:c0 + WINDOW].mean():.1f}%")
print()
print("  %-12s %12s %14s" % ("target", "body NLL", "struct pen"))
for name, bj, pj in zip(TARGETS, body, pen):
    print("  %-12s %12.6f %14.6f" % (name, float(bj), float(pj)))
print("  %-12s %12.6f %14.6f" % ("SUM", b_sum, p_sum))
print()
print(f"  sum gate BCE (pw {POS_WEIGHT})  {g_sum:.6f}")
print(f"  balancer output        {balanced:.6f}   (= 0.5*body + 1.0*gate)")
print(f"  weighted penalty       {w_pen:.6f}")
print()
print(f"  SWEEP over {DRAWS} stochastic dropout draws (dropout {cfg.get('dropout_rate')} ON):")
print("  %-6s %10s %10s %10s %10s %8s" % ("stat", "body", "gate", "pen", "balancer", "share%"))
for lab, v in (
    ("mean", sweep.mean(0)),
    ("sd", sweep.std(0, ddof=1)),
    ("min", sweep.min(0)),
    ("max", sweep.max(0)),
):
    print("  %-6s %10.6f %10.6f %10.6f %10.6f %7.1f%%" % (lab, *v))
print("  %-6s %10.6f %10.6f %10.6f %10.6f %7.1f%%" % ("eval", *eval_row))
print(f"\n  ⇒ share of the loss the optimiser sees: {sweep[:, 4].min():.1f}% .. "
      f"{sweep[:, 4].max():.1f}%  (dropout off: {eval_row[4]:.1f}%)")
print()
print("  ⚠️ AT INITIALISATION on this same window and seed (struct_weight_derivation_v3.py), the")
print("     control's figures were: body 1.247872, gate 2.079120, penalty 0.827130,")
print("     balancer 2.703056, share 11.3%. Compare term for term — a LARGE move in the penalty's")
print("     share is the C-325 drift §3c could not otherwise see, and a reason to publish the")
print("     per-term trajectory column (C-365) before the NEXT arm.")
print()
print("  ⛔ TWO POINTS, NOT A TRAJECTORY. This cannot show when a term drifted, a spike that")
print("     resolved, or a divergence that reversed. C-325 is narrowed, not retired.")
print("\n" + "=" * 94)
print("COST, measured")
print("=" * 94)
print("  wall time %.2f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
