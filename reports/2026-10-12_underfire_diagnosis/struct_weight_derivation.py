"""Derive `struct_penalty_weight` by gradient-norm matching at initialisation.

§3 of `reports/2026-10-21_structure_penalty_arm_prereg.md`, which commits to this rather than to a
sweep: measure the two terms' gradient norms on the real vehicle at initialisation and set the
weight so they are equal. ⛔ One measurement, no money, and it removes a free parameter rather than
multiplying the arm by a sweep.

  weight = ||d(body NLL)/d(theta)|| / ||d(struct_pen)/d(theta)||

⚠️ This is GradNorm's idea as a ONE-TIME CONSTANT, not as the dynamic algorithm, which was declined
on cost. The constant is measured once, reported, and then fixed for the run.

⛔ PRE-COMMITTED FALLBACK (§3): if the derived weight makes the term exceed 50% of the total
loss at initialisation, halve it until it does not, and report that this happened.

⚠️ AT INITIALISATION, which is a real limitation and is the one C-325 is about: a quantity measured
on an untrained network can mislead about a trained one (two SCREEN-2 mechanism tests were recorded
as ruling a mechanism out, measured at init, and the arm's gradient did explode 33 lessons later).
The weight is therefore a STARTING SCALE, not a claim that the two terms stay balanced — and the
run's per-task trajectory log is what would show them drifting apart.

Builds a FRESH model from the delivered artifact's own config sidecar (so the architecture, channel
counts and family match the control exactly) and feeds one real curriculum-sized window from the
cached land pull. No training, no pod, CPU. Cost printed.
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import SB, build_volume, pull  # noqa: E402

SP = Path(__file__).parent
WINDOW = 32  # the curriculum's window size, and divisible by both _STRUCT_BLOCKS
t_start = time.time()

ART = sys.argv[1] if len(sys.argv) > 1 else None
if ART is None:
    raise SystemExit("usage: struct_weight_derivation.py <path to calibration_model_*.pt>")

cfg = json.loads(Path(f"{ART}.config.json").read_text())
print("=" * 92)
print("DERIVING struct_penalty_weight BY GRADIENT-NORM MATCHING AT INITIALISATION")
print("=" * 92)
print(f"  config sidecar   {Path(ART).name}.config.json")
print(f"  family           {cfg['output_distribution']}")
print(f"  hidden channels  {cfg.get('total_hidden_channels')}")

from views_hydranet.distributions import get_family  # noqa: E402
from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402
from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate  # noqa: E402
from views_hydranet.train.training_engine import (  # noqa: E402
    _STRUCT_BLOCKS,
    _structure_penalty,
)
from views_hydranet.utils.utils import choose_model  # noqa: E402

ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
device = torch.device("cpu")
model = choose_model(cfg, device)          # FRESH weights — this is the "at initialisation" part
model.train()
family = get_family(cfg["output_distribution"])
criterion = FamilyLoss(family)
npar = criterion.n_params

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
vol, land, _pg = build_volume(df, list(range(421, 423)))
# One window, chosen by LAND COVERAGE only — never by activity. The population slot of C-351: a
# window picked for conflict would set the weight from the busiest patch on the planet.
rows = np.where(land.any(axis=1))[0]
cols = np.where(land.any(axis=0))[0]
r0 = int(rows[len(rows) // 2]) - WINDOW // 2
c0 = int(cols[len(cols) // 2]) - WINDOW // 2
x = vol[:, 0, :, r0 : r0 + WINDOW, c0 : c0 + WINDOW]
y = vol[:, 1, :, r0 : r0 + WINDOW, c0 : c0 + WINDOW]
print(f"  window           rows {r0}-{r0 + WINDOW}, cols {c0}-{c0 + WINDOW}, "
      f"land {100 * land[r0:r0 + WINDOW, c0:c0 + WINDOW].mean():.1f}%, "
      f"positive cells {int((y[0, SB] > 0).sum())}")
print(f"  block sizes      {_STRUCT_BLOCKS}")

h = torch.zeros(1, cfg["total_hidden_channels"], WINDOW, WINDOW)
out = model(x, h)
params = out.reg[:, SB * npar : (SB + 1) * npar].permute(0, 2, 3, 1)
target = y[:, SB]
gate_logits = out.cls[:, SB]


def grad_norm(term: torch.Tensor) -> float:
    model.zero_grad(set_to_none=True)
    term.backward(retain_graph=True)
    sq = sum(
        float((p.grad.detach() ** 2).sum()) for p in model.parameters() if p.grad is not None
    )
    return sq**0.5


nll = criterion(params, target)
pen = _structure_penalty(params[..., 0], gate_logits, target)
g_nll = grad_norm(nll)
g_pen = grad_norm(pen)

print("\n  %-28s %14s %14s" % ("term", "value", "grad norm"))
print("  %-28s %14.6f %14.6e" % ("body NLL", float(nll), g_nll))
print("  %-28s %14.6f %14.6e" % ("structure penalty", float(pen), g_pen))

if g_pen == 0.0:
    raise SystemExit(
        "VOID: the structure penalty has zero gradient at initialisation — it cannot be balanced, "
        "and an arm run at any weight would be inert (C-324)."
    )

weight = g_nll / g_pen
share = weight * float(pen) / (float(nll) + weight * float(pen))
print("\n  ⇒ derived weight (grad-norm matched)   %.6g" % weight)
print("    the term's share of the total loss   %.1f%%" % (100 * share))

halvings = 0
while share > 0.5:
    weight /= 2.0
    halvings += 1
    share = weight * float(pen) / (float(nll) + weight * float(pen))
if halvings:
    print("  ⛔ §3's FALLBACK FIRED: halved %d time(s) to keep the term under 50%% of the loss"
          % halvings)
    print("    final weight                        %.6g" % weight)
    print("    final share                         %.1f%%" % (100 * share))
else:
    print("    §3's 50% fallback did not fire")

print("""
  ⚠️ READ THIS AS A STARTING SCALE, NOT A BALANCE CLAIM. Measured on an UNTRAINED network (C-325:
  two mechanism tests measured at init were recorded as ruling a mechanism out, and the arm's
  gradient exploded 33 lessons later). One window, one seed, one step. The run's per-task
  trajectory log is what would show the two terms drifting apart.""")

print("\n" + "=" * 92)
print("COST, measured")
print("=" * 92)
print("  wall time %.2f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2))
