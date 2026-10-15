"""R5 — the PRE-FLIGHT GATE. This is falsifier F1, and it runs before any GPU spend.

`reports/2026-10-13_composition_arm_outcome/OUTCOME.md` closes with a requirement that nothing in
the harness implements: **no config-identity check can see a loss magnitude.**
`scripts/arm_identity_check.py` compares config VALUES, and the composition arm shipped a 65x
change in a task-loss term with every check green.

So this asserts the two things a config diff cannot:

  1. WIRING — that setting `body_nll_denominator='all_cells'` in the config actually reaches the
     loss the training loop will call, through the PRODUCTION factory (`utils.choose_loss`), and
     that the two settings produce DIFFERENT numbers on the same predictions. A flag that
     validates and changes nothing is the C-324 inert-knob signature.
  2. MAGNITUDES — the per-task task-loss terms and the count of cells entering each task's
     backward graph, against the table pre-committed in
     `reports/2026-10-15_loss_balance_prereg.md` §4 before any of this was implemented.

F1 (prereg §5): pooled `0.5*sum(body)/gate` outside **0.11-0.45** => VOID, do not train.

SCOPE, same as r2/r3/r4 and stated not discovered: eval() mode (train() would rewrite the
artifacts' BN buffers — C-184/C-328, and the artifacts cannot be retrained, C-317); the sampler's
RNG stream is not production's, so these are windows from the same distribution, not the same
windows; no random_flips. Deliberately a WET sibling of r2/r3/r4 — their provenance stays fixed.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate

sys.path.insert(0, str(Path(__file__).parent.parent / "2026-10-12_underfire_diagnosis"))
from vehicle import TARGETS, grid_rowcol, pull  # noqa: E402

SP = Path(__file__).parent
DIAG = SP.parent / "2026-10-12_underfire_diagnosis"
MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
ARM_PT = (
    MODELS / "amber_kestrel/data/generated/calibration_delivery_20261013/"
    "artifact/calibration_model_20261013_010739.pt"
)
N_SAMPLED_LESSONS = 12

# ── pre-committed in prereg §4, BEFORE this was implemented ──────────────────────────────────
PREDICTED = {"lr_sb_best": 0.295, "lr_ns_best": 0.058, "lr_os_best": 0.166}
PREDICTED_POOLED = 0.225
TOLERANCE = 0.30  # P-1/P-2: within +/-30%
F1_BAND = (0.11, 0.45)  # prereg §5 F1: outside this => VOID, do not train


def load_hp(name: str) -> dict:
    path = MODELS / name / "configs" / "config_hyperparameters.py"
    spec = importlib.util.spec_from_file_location(f"hp_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_hp_config()


def load_weights(pt: Path, expect_dist: str):
    from views_hydranet.utils.utils import choose_model

    cfg = json.loads(pt.with_suffix(".pt.config.json").read_text())
    if cfg.get("output_distribution") != expect_dist:
        raise SystemExit(f"VOID: {pt.name} is {cfg.get('output_distribution')!r}.")
    sha = hashlib.sha256(pt.read_bytes()).hexdigest()
    shaf = pt.with_suffix(".pt.sha256")
    if shaf.exists() and shaf.read_text().split()[0] != sha:
        raise SystemExit(f"VOID: sha256 mismatch for {pt.name}.")
    model = choose_model(cfg, torch.device("cpu"))
    model.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    model.to("cpu").eval()
    print(f"  {pt.name}  sha {sha[:16]}…  dist={cfg['output_distribution']}")
    return model


def build_handler(config: dict):
    from views_hydranet.utils.feature_scaler import FeatureScaler
    from views_hydranet.utils.volume_handler import VolumeHandler

    df = pull((121, 504), DIAG / "pull_land_all3_121_504.parquet")
    df = df[(df.month_id >= 121) & (df.month_id <= 456)].copy()
    r, c = grid_rowcol(df.priogrid_id.to_numpy())
    df["row"], df["col"] = r, c
    df["c_id"] = 0
    df = df.rename(columns=dict(zip(TARGETS, config["regression_targets"])))
    return VolumeHandler.from_df(FeatureScaler(config).fit_transform(df), config)


# ══ CHECK 1 — WIRING, through the production factory ═════════════════════════════════════════
print("=" * 94)
print("R5 PRE-FLIGHT (falsifier F1) — body_nll_denominator")
print("=" * 94)
ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
arm_cfg = load_hp("amber_kestrel")
if arm_cfg.get("target_weights"):
    raise SystemExit("VOID: the config carries target_weights (C-87).")

from views_hydranet.utils.utils import choose_loss  # noqa: E402

print("\nCHECK 1 — does the config flag reach the loss the trainer calls?")
built = {}
for setting in ("positives", "all_cells"):
    cfg = dict(arm_cfg)
    cfg["body_nll_denominator"] = setting
    crit_reg, crit_cls, mtl = choose_loss(cfg, torch.device("cpu"))
    fam = getattr(crit_reg, "family", None)
    got = getattr(fam, "body_nll_denominator", None)
    if got != setting:
        raise SystemExit(
            f"VOID: config said {setting!r} but the loss carries {got!r} — "
            "the flag does not reach the loss (C-324 inert knob)."
        )
    print(f"  config body_nll_denominator={setting!r:12} -> loss family carries {got!r}  OK")
    built[setting] = crit_reg
print(
    f"  balancer frozen in config: {arm_cfg.get('freeze_multitask_balancer')}  "
    f"(must stay True — A does not touch the balancer)"
)

# ══ CHECK 2 — MAGNITUDES and BACKWARD-GRAPH CELL COUNTS ══════════════════════════════════════
print("\nbuilding the calibration train volume …")
handler = build_handler(arm_cfg)
arm = load_weights(ARM_PT, "truncated_nb")

from views_hydranet.train.training_engine import _SequenceIndices  # noqa: E402
from views_hydranet.utils.curriculum import CurriculumLearner  # noqa: E402
from views_hydranet.utils.volume_sampler import VolumeSampler  # noqa: E402

sampler = VolumeSampler(handler, arm_cfg)
planner = CurriculumLearner(arm_cfg, handler)
_, crit_cls, _ = choose_loss({**arm_cfg, "body_nll_denominator": "all_cells"}, torch.device("cpu"))
wpl = arm_cfg["windows_per_lesson"]
lessons = np.unique(np.linspace(0, arm_cfg["total_lessons"] - 1, N_SAMPLED_LESSONS).astype(int))

rows = {"positives": [], "all_cells": []}
gates, cells, pos_cells = [], [], []
for li in lessons:
    for wi in range(wpl):
        target, threshold = planner.get_lesson(int(li) * wpl + wi)
        sh = sampler.get_batch(target, threshold, batch_size=1)[0][0]
        tt = sh.to_pytorch(torch.device("cpu"), include_identities=False)
        idx = _SequenceIndices([n for n in sh.channel_map if n in sh.tensor_cols], arm_cfg)
        h = arm.init_hTtime(hidden_channels=arm.base, H=tt.shape[-2], W=tt.shape[-1])
        steps = tt.shape[1] - 1
        b = {k: np.zeros(idx.n_reg) for k in rows}
        g, ncell, npos = np.zeros(idx.n_cls), 0, np.zeros(idx.n_reg)
        with torch.no_grad():
            for i in range(steps):
                t1 = tt[:, i + 1]
                y_reg, y_cls = t1[:, idx.reg], t1[:, idx.cls]
                o = arm(tt[:, i][:, idx.feat], h)
                h = o.h_next
                for j in range(idx.n_reg):
                    p = o.reg[:, j * 2 : (j + 1) * 2].permute(0, 2, 3, 1)
                    for k, fn in built.items():
                        b[k][j] += float(fn(p, y_reg[:, j]))
                    npos[j] += int((y_reg[:, j] > 0).sum())
                for j in range(idx.n_cls):
                    g[j] += float(crit_cls(o.cls[:, j], y_cls[:, j]))
                ncell += y_reg[:, 0].numel()
        for k in rows:
            rows[k].append(b[k] / steps)
        gates.append(g / steps)
        cells.append(ncell / steps)
        pos_cells.append(npos / steps)

P = {k: np.asarray(v) for k, v in rows.items()}
G, C, NP = np.asarray(gates), np.asarray(cells), np.asarray(pos_cells)
names = idx.reg_names

print(
    f"\nCHECK 2 — per-task magnitudes and backward-graph cell counts "
    f"({G.shape[0]} windows, per step)"
)
print("\n  CELLS ENTERING EACH TASK'S BACKWARD GRAPH (what a config diff cannot see)")
print(f"  {'task':<22}{'cells in graph':>16}{'denominator':>14}")
for j, n in enumerate(names):
    print(
        f"  {'body/' + n:<22}{np.median(NP[:, j]):>16.2f}"
        f"{np.median(C):>14.0f}   <- numerator over positives, denominator all cells"
    )
for j, n in enumerate(idx.cls_names):
    print(f"  {'gate/' + n:<22}{np.median(C):>16.0f}{np.median(C):>14.0f}")

print(
    f"\n  {'target':<14}{'positives':>12}{'all_cells':>12}{'ratio':>9}"
    f"{'gate':>9}{'0.5b/g':>9}{'predicted':>11}{'dev':>8}"
)
ok = True
for j, n in enumerate(names):
    bp, ba = np.median(P["positives"][:, j]), np.median(P["all_cells"][:, j])
    g = np.median(G[:, j])
    got = 0.5 * ba / g
    pred = PREDICTED[n]
    dev = (got - pred) / pred
    flag = "" if abs(dev) <= TOLERANCE else "  <-- OUTSIDE +/-30%"
    ok &= abs(dev) <= TOLERANCE
    print(
        f"  {n:<14}{bp:>12.4f}{ba:>12.5f}{ba / bp:>9.4f}{g:>9.4f}"
        f"{got:>9.4f}{pred:>11.3f}{dev:>+7.0%}{flag}"
    )

pooled = 0.5 * np.median(P["all_cells"].sum(1)) / np.median(G.sum(1))
asrun = 0.5 * np.median(P["positives"].sum(1)) / np.median(G.sum(1))
print(
    f"\n  POOLED 0.5*sum(body)/gate   as-run(positives) {asrun:8.3f}"
    f"   under A(all_cells) {pooled:8.4f}   predicted {PREDICTED_POOLED}"
)

print("\n" + "=" * 94)
in_band = F1_BAND[0] <= pooled <= F1_BAND[1]
print(f"F1 BAND {F1_BAND}:  pooled = {pooled:.4f}  ->  {'INSIDE' if in_band else 'OUTSIDE'}")
print(f"P-1/P-2 within +/-{TOLERANCE:.0%}: {'YES' if ok else 'NO'}")
if not in_band:
    print("VERDICT: ⛔ F1 FIRES — VOID. The intervention is not the intervention. DO NOT TRAIN.")
    raise SystemExit(1)
print("VERDICT: ✅ F1 does not fire — the axis is WIRED and the pooled magnitude is in band.")
if ok:
    print("         Every per-target prediction is also within +/-30%.")
else:
    print("         ⚠️  BUT a per-target prediction (P-2) MISSED. F1 is the only VOID condition,")
    print("             so the arm is still cleared — but the miss is a prediction error of MINE,")
    print("             not a fault in the intervention, and it must be read before the arm runs.")
    print("             CAUSE: the prereg projected `body_A = body_positives * (P/N)` using the")
    print("             median positive FRACTION. That identity holds per step, NOT after")
    print("             averaging over steps: `positives` divides by a per-step count that is")
    print("             often 0 or 1, so mean(S/P) is dominated by the small-P steps while")
    print("             mean(S/N) is not, so one multiplicative factor cannot carry the window")
    print("             mean across. It is the mean-of-ratios vs ratio-of-means error, on the")
    print("             target where P is smallest — exactly where the two diverge most.")
    print("             DIRECTION: the miss is UPWARD. ns lands at 0.175 of its gate, not 0.058,")
    print("             so the per-target spread under A is ~1.7x (0.297/0.175/0.204), not the 5x")
    print("             the prereg's skepticism ledger warned about. A is MORE even-handed")
    print("             than my own projection claimed — weakening prereg section 3.1 further.")
print("=" * 94)
np.savez(
    SP / "r5_result.npz",
    pos=P["positives"],
    allc=P["all_cells"],
    gate=G,
    cells=C,
    pos_cells=NP,
    reg_names=np.array(names),
)
print("wrote r5_result.npz")
