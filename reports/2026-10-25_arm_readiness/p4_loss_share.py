"""P4: the penalty's share of the loss, against the denominator the OPTIMISER actually sees.

⛔ WHY THIS EXISTS. `2026-10-21_structure_penalty_arm_prereg.md` §3c reports the structure penalty
at **21.7% of the loss** and reads §3's pre-committed 50%-of-loss fallback against that figure. The
21.7% divides by the **body NLL sum alone**. The loss the optimiser minimises is the BALANCER's
output plus the penalty — `loss = multitaskloss_instance(losses)` at `training_engine.py:1120`,
then `+ struct_penalty_weight * struct_pen` at `:1128` — i.e.
`0.5*sum(body) + 1.0*sum(gate) + w*pen`, because the frozen balancer gives every regression
task `1/((is_regression+1)*e^0) = 0.5` and every classification task 1.0.

⭐ The finding is SOFT and the distinction is the point: every candidate denominator puts the share
under 50%, so the fallback does not fire under any reading and **the pinned weight 0.416904 is
unchanged**. The reported number is wrong; the decision it gates is not.

⚠️ It also prints the 2x ambiguity in §3's own rule. §3 says match `||d(body_NLL)/d(theta)||`. Read
literally — what v1, v2 and v3 all did — the weight is 0.416904. Read as "the body term as the loss
actually contains it", the body enters at x0.5 and the matched weight is 0.208452. The rule is NOT
changed here: amending a pre-registered rule after seeing the number is worse than the ambiguity.

Same vehicle, window, seed and fresh weights as `struct_weight_derivation_v3.py`, so the body and
penalty figures below are directly comparable to §3c's table. CPU, no training, seconds.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
SP = REPO / "reports/2026-10-12_underfire_diagnosis"
sys.path.insert(0, str(REPO))  # ⛔ FIRST: the working tree must win over installed 0.1.5 (C-347)
sys.path.insert(1, str(SP))

from vehicle import build_volume, pull  # noqa: E402

from views_hydranet.distributions import get_family  # noqa: E402
from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402
from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate  # noqa: E402
from views_hydranet.train.training_engine import _structure_penalty  # noqa: E402
from views_hydranet.utils.mtloss import MultiTaskLoss  # noqa: E402
from views_hydranet.utils.utils import choose_model  # noqa: E402

ART = (
    "/home/simon/views-diagnostics/hf_control_artifact/artifact/"
    "calibration_model_20261025_132124.pt"
)
WEIGHT = 0.416904  # §3c's pinned value, the thing under test
WINDOW = 32

cfg = json.loads(Path(f"{ART}.config.json").read_text())
ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
model = choose_model(cfg, torch.device("cpu"))
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
out = model(x, torch.zeros(1, cfg["total_hidden_channels"], WINDOW, WINDOW))

body, pen = [], []
for j in range(3):
    params_j = out.reg[:, j * npar : (j + 1) * npar].permute(0, 2, 3, 1)
    body.append(criterion(params_j, y[:, j]))
    pen.append(_structure_penalty(family, params_j, out.cls[:, j], y[:, j]))

# The gate terms as training builds them: weighted BCE at the roster's loss_class_pos_weight 2.0.
pos_weight = torch.tensor(2.0)
gate = [
    torch.nn.functional.binary_cross_entropy_with_logits(
        out.cls[:, j], (y[:, j] > 0).float(), pos_weight=pos_weight
    )
    for j in range(3)
]

losses = torch.stack(body + gate)  # the 6-vector the balancer combines, body first
mtl = MultiTaskLoss(
    is_regression=torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), reduction="sum"
)
balanced = float(mtl(losses))  # log_vars frozen at their zero initialisation

b_sum, p_sum, g_sum = float(sum(body)), float(sum(pen)), float(sum(gate))
weighted_pen = WEIGHT * p_sum

print("=" * 88)
print("P4 — the penalty's share of the loss, against THREE candidate denominators")
print("=" * 88)
print(f"  views_hydranet            {Path(__import__('views_hydranet').__file__)}")
print(f"  vehicle                   {Path(ART).name}  ({cfg['output_distribution']})")
print(f"  sum body NLL              {b_sum:.6f}")
print(f"  sum gate BCE (pw 2.0)     {g_sum:.6f}")
print(f"  sum structure penalty     {p_sum:.6f}")
print(f"  balancer output           {balanced:.6f}")
print(f"    = 0.5*body + 1.0*gate   {0.5 * b_sum + g_sum:.6f}   (the frozen-balancer identity)")
print(f"  weighted penalty w*pen    {weighted_pen:.6f}   at w = {WEIGHT}")
print()
print("  %-52s %8s" % ("denominator", "share"))
for label, den in [
    ("sum(body) + w*pen          <- §3c REPORTS THIS", b_sum + weighted_pen),
    ("balancer  + w*pen          <- the optimiser sees this", balanced + weighted_pen),
    ("0.5*sum(body) + w*pen      <- body only, as weighted", 0.5 * b_sum + weighted_pen),
]:
    print("  %-52s %7.1f%%" % (label, 100 * weighted_pen / den))
print()
print("  ⇒ §3's 50% fallback does NOT fire under ANY reading, so the pinned weight is unchanged.")
print(f"  ⚠️ If §3 is read as matching the body AS WEIGHTED (x0.5): w = {0.5 * WEIGHT:.6f}")
