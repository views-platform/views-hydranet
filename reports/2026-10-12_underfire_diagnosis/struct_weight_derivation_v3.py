"""Re-derive `struct_penalty_weight` over ALL THREE targets, on the vehicle that will train.

⛔ WHY v3 EXISTS, and it is a correctness fix rather than a refinement. v1/v2 measured the
numerator and denominator on **state-based alone** (`SB`, one call to `_structure_penalty`). The
training loop does not. `training_engine.py:995` accumulates

    struct_pen = Σ_j tw_j * _structure_penalty(family_j, pred_j, gate_logits_j, target_j)

over every regression target, and `:1127-1128` multiplies that **sum** by the single scalar
`struct_penalty_weight`. ⇒ A weight matched against one target's gradient and applied to three
lands the term at the wrong strength. §3 of `reports/2026-10-21_structure_penalty_arm_prereg.md`
commits to equalising the two gradient norms; v3 equalises them for **the objects that actually
appear in the loss**.

⛔ AND THERE IS NO PER-VIOLENCE-TYPE WEIGHT. `tw_j` comes from `target_weights`, which is
`Field(default=None)` and which `heavy_freighter` does not set, so `tw_j = 1.0` for all three
(`training_engine.py:927-929`). The three targets therefore receive EQUAL structural pressure
despite positive rates of 0.009465 / 0.000861 / 0.004783 — an ~11x spread. That is a property of
the arm as designed, recorded here, not something this script changes.

⛔ IMPORT PATH. v1 failed with `_structure_penalty() takes 3 positional arguments but 4 were
given`: run from this subdirectory, the INSTALLED `views_hydranet` 0.1.5 (3-arg) shadowed the
working tree (4-arg). The env is a regular install, not a dev install — there is no editable
finder. The repo root is therefore prepended to `sys.path` BELOW, and the module's resolved
`__file__` is printed so the reader can see which one answered.
⚠️ `git diff v0.1.7 -- views_hydranet/train/training_engine.py` is EMPTY, so the working tree is
byte-identical to the 0.1.7 tag for this file: the 4-arg signature is 0.1.7's, not a local edit.

Everything else follows v2: a FRESH model from the delivered artifact's own config sidecar, one
real curriculum-sized window chosen by LAND COVERAGE ONLY (never by activity — C-351's population
slot), CPU, no training, cost printed. The parquet cache is the same file v2 used, deliberately, so
the only thing that moved between the two numbers is the target set and the vehicle.
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).parent
REPO = SP.parents[1]
sys.path.insert(0, str(REPO))      # ⛔ FIRST: the working tree must win over installed 0.1.5
sys.path.insert(1, str(SP))

from vehicle import TARGETS, build_volume, pull  # noqa: E402

WINDOW = 32  # the curriculum's window size, and divisible by both _STRUCT_BLOCKS
t_start = time.time()

ART = sys.argv[1] if len(sys.argv) > 1 else None
if ART is None:
    raise SystemExit("usage: struct_weight_derivation_v3.py <path to calibration_model_*.pt>")

cfg = json.loads(Path(f"{ART}.config.json").read_text())

import views_hydranet  # noqa: E402
from views_hydranet.distributions import get_family  # noqa: E402
from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402
from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate  # noqa: E402
from views_hydranet.train.training_engine import (  # noqa: E402
    _STRUCT_BLOCKS,
    _structure_penalty,
)
from views_hydranet.utils.utils import choose_model  # noqa: E402

print("=" * 92)
print("DERIVING struct_penalty_weight BY GRADIENT-NORM MATCHING — ALL THREE TARGETS")
print("=" * 92)
print(f"  views_hydranet   {views_hydranet.__file__}")
print(f"  config sidecar   {Path(ART).name}.config.json")
print(f"  family           {cfg['output_distribution']}")
print(f"  dropout_rate     {cfg.get('dropout_rate')}")
print(f"  hidden channels  {cfg.get('total_hidden_channels')}")
print(f"  target_weights   {cfg.get('target_weights')!r}   "
      f"(None ⇒ tw = 1.0 for every target, i.e. NO per-violence-type weight)")
print(f"  reg targets      {cfg.get('regression_targets')}")

ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
device = torch.device("cpu")
model = choose_model(cfg, device)          # FRESH weights — this is the "at initialisation" part
model.train()
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
print(f"  window           rows {r0}-{r0 + WINDOW}, cols {c0}-{c0 + WINDOW}, "
      f"land {100 * land[r0:r0 + WINDOW, c0:c0 + WINDOW].mean():.1f}%")
print(f"  block sizes      {_STRUCT_BLOCKS}")

h = torch.zeros(1, cfg["total_hidden_channels"], WINDOW, WINDOW)
out = model(x, h)


def grad_norm(term: torch.Tensor) -> float:
    model.zero_grad(set_to_none=True)
    term.backward(retain_graph=True)
    sq = sum(
        float((p.grad.detach() ** 2).sum()) for p in model.parameters() if p.grad is not None
    )
    return sq**0.5


# ⛔ Per target FIRST, because the per-target numbers are the finding: they say how much of the
# summed penalty each violence type contributes under the equal tw = 1.0 the config implies.
print("\n  %-10s %10s %14s %14s %14s %14s"
      % ("target", "pos cells", "body NLL", "‖∂NLL‖", "struct pen", "‖∂pen‖"))
nll_terms, pen_terms = [], []
for j, name in enumerate(TARGETS):
    params_j = out.reg[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1)
    target_j = y[:, j]
    nll_j = criterion(params_j, target_j)
    pen_j = _structure_penalty(family, params_j, out.cls[:, j], target_j)
    nll_terms.append(nll_j)
    pen_terms.append(pen_j)
    print("  %-10s %10d %14.6f %14.6e %14.6f %14.6e"
          % (name, int((target_j[0] > 0).sum()), float(nll_j), grad_norm(nll_j),
             float(pen_j), grad_norm(pen_j)))

# The objects the loss actually contains: the SUM over targets, each at tw = 1.0.
nll_tot = sum(nll_terms)
pen_tot = sum(pen_terms)
g_nll = grad_norm(nll_tot)
g_pen = grad_norm(pen_tot)

print("\n  %-28s %14s %14s" % ("summed over 3 targets", "value", "grad norm"))
print("  %-28s %14.6f %14.6e" % ("body NLL", float(nll_tot), g_nll))
print("  %-28s %14.6f %14.6e" % ("structure penalty", float(pen_tot), g_pen))

if g_pen == 0.0:
    raise SystemExit(
        "VOID: the structure penalty has zero gradient at initialisation — it cannot be balanced, "
        "and an arm run at any weight would be inert (C-324)."
    )

weight = g_nll / g_pen
share = weight * float(pen_tot) / (float(nll_tot) + weight * float(pen_tot))
print("\n  ⇒ derived weight (grad-norm matched)   %.6g" % weight)
print("    the term's share of the total loss   %.1f%%" % (100 * share))

halvings = 0
while share > 0.5:
    weight /= 2.0
    halvings += 1
    share = weight * float(pen_tot) / (float(nll_tot) + weight * float(pen_tot))
if halvings:
    print("  ⛔ §3's FALLBACK FIRED: halved %d time(s) to keep the term under 50%% of the loss"
          % halvings)
    print("    final weight                        %.6g" % weight)
    print("    final share                         %.1f%%" % (100 * share))
else:
    print("    §3's 50% fallback did not fire")

print("\n  ⚠️ SUPERSEDED, for comparison. The 2026-10-21 derivations were SB-only AND were run on")
print("     calibration_model_20261013_010739.pt = the truncated_nb vehicle 2a5efc2a -- NOT")
print("     the nb vehicle the arm runs on:")
print("       v1  17.7925 -> 8.89624 after one halving, term at 38.2% of the loss")
print("       v2  27.0768, no halving, term at 44.9%")

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
