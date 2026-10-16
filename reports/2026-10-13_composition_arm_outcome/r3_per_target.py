"""R3 — the same measurement as R2, split by target. R2 pooled the three, which hid a second axis.

WHY THIS EXISTS. R2 reported `sum over the three regression targets` and a single pooled positive
fraction. But `training_engine.py:669-760` appends ONE loss per target, and the truncated family
normalises each target by ITS OWN positive count (`nb_core.py:77-79`). sb/ns/os have very
different positive rates, so the per-target amplification differs — and `target_weights` is None
in both configs (C-87), so nothing compensates. R2's pooled number cannot see that.

Deliberately a sibling of `r2_loss_scale.py` rather than an edit to it: R2's numbers are already
cited in OUTCOME.md, and its provenance should stay fixed. The loaders are duplicated on purpose
(WET), the sampling is identical, and the pooled column is recomputed here as a cross-check
against R2 — if the sum of the per-target terms does not reproduce R2, one of the two is wrong.

SCOPE, same as R2: eval mode (dropout off, BN from buffers — `train()` would rewrite the
artifacts' buffers, C-184/C-328); the sampler's RNG stream is not the production one, so the
windows are drawn from the same distribution but are not the same windows; no `random_flips`.
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
CONTROL_PT = Path("/home/simon/Documents/views_hydranet_vehicle_20261012/"
                  "calibration_model_20261012_033112.pt")
ARM_PT = (MODELS / "amber_kestrel/data/generated/calibration_delivery_20261013/"
          "artifact/calibration_model_20261013_010739.pt")
N_SAMPLED_LESSONS = 12
POS_WEIGHT = 2.0


def load_hp(name: str) -> dict:
    path = MODELS / name / "configs" / "config_hyperparameters.py"
    spec = importlib.util.spec_from_file_location(f"hp_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_hp_config()


def load_weights(pt: Path, expect_dist: str):
    from views_hydranet.utils.utils import choose_model

    sidecar = pt.with_suffix(".pt.config.json")
    cfg = json.loads(sidecar.read_text())
    if cfg.get("output_distribution") != expect_dist:
        raise SystemExit(f"VOID: {pt.name} is {cfg.get('output_distribution')!r}, "
                         f"expected {expect_dist!r}.")
    sha = hashlib.sha256(pt.read_bytes()).hexdigest()
    shaf = pt.with_suffix(".pt.sha256")
    if shaf.exists() and shaf.read_text().split()[0] != sha:
        raise SystemExit(f"VOID: sha256 mismatch for {pt.name}.")
    model = choose_model(cfg, torch.device("cpu"))
    model.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    model.hurdle_nb_theta = cfg.get("hurdle_nb_theta")
    model.hurdle_lognormal_sigma = cfg.get("hurdle_lognormal_sigma")
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


def per_target(model, config, handler, families: dict):
    """-> {family: [W, n_reg]}, gate [W, n_cls], pos_frac [W, n_reg]."""
    from views_hydranet.distributions.family_loss import FamilyLoss
    from views_hydranet.train.training_engine import _SequenceIndices
    from views_hydranet.utils.curriculum import CurriculumLearner
    from views_hydranet.utils.volume_sampler import VolumeSampler
    from views_hydranet.utils.weighted_bce_loss import WeightedBCEWithLogitsLoss

    sampler = VolumeSampler(handler, config)
    planner = CurriculumLearner(config, handler)
    gate_fn = WeightedBCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    body = {k: FamilyLoss(v) for k, v in families.items()}
    npar = {v.n_params for v in body.values()}
    if len(npar) != 1:
        raise SystemExit(f"n_params differ: {npar}")
    npar = npar.pop()

    wpl = config["windows_per_lesson"]
    lessons = np.unique(np.linspace(0, config["total_lessons"] - 1, N_SAMPLED_LESSONS).astype(int))
    out = {k: [] for k in body}
    gates, pfs = [], []
    for li in lessons:
        for wi in range(wpl):
            target, threshold = planner.get_lesson(int(li) * wpl + wi)
            sh = sampler.get_batch(target, threshold, batch_size=1)[0][0]
            tt = sh.to_pytorch(torch.device("cpu"), include_identities=False)
            idx = _SequenceIndices([n for n in sh.channel_map if n in sh.tensor_cols], config)
            h = model.init_hTtime(hidden_channels=model.base, H=tt.shape[-2], W=tt.shape[-1])
            steps = tt.shape[1] - 1
            b = {k: np.zeros(idx.n_reg) for k in body}
            g = np.zeros(idx.n_cls)
            pos = np.zeros(idx.n_reg)
            n = 0
            with torch.no_grad():
                for i in range(steps):
                    t1 = tt[:, i + 1]
                    y_reg, y_cls = t1[:, idx.reg], t1[:, idx.cls]
                    o = model(tt[:, i][:, idx.feat], h)
                    h = o.h_next
                    for j in range(idx.n_reg):
                        p = o.reg[:, j * npar:(j + 1) * npar].permute(0, 2, 3, 1)
                        for k, fn in body.items():
                            b[k][j] += float(fn(p, y_reg[:, j]))
                        pos[j] += int((y_reg[:, j] > 0).sum())
                    for j in range(idx.n_cls):
                        g[j] += float(gate_fn(o.cls[:, j], y_cls[:, j]))
                    n += y_reg[:, 0].numel()
            for k in body:
                out[k].append(b[k] / steps)
            gates.append(g / steps)
            pfs.append(pos / n)
    return ({k: np.asarray(v) for k, v in out.items()},
            np.asarray(gates), np.asarray(pfs), idx.reg_names, idx.cls_names)


ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
print("loading vehicles …")
ctrl = load_weights(CONTROL_PT, "nb")
arm = load_weights(ARM_PT, "truncated_nb")
ctrl_cfg, arm_cfg = load_hp("violet_visitor"), load_hp("amber_kestrel")
if ctrl_cfg.get("target_weights") or arm_cfg.get("target_weights"):
    raise SystemExit("VOID: a config carries target_weights — the per-target reading below "
                     "assumes C-87's weights are all 1.0.")

print("building the calibration train volume …")
handler = build_handler(ctrl_cfg)

from views_hydranet.distributions import resolve_family  # noqa: E402

print("control weights, both families …")
cb, cg, pf, reg_names, cls_names = per_target(
    ctrl, ctrl_cfg, handler,
    {"nb": resolve_family("nb"), "trunc": resolve_family("truncated_nb")})
print("arm weights …")
ab, ag, apf, _, _ = per_target(arm, arm_cfg, handler,
                               {"trunc": resolve_family("truncated_nb")})

W = cg.shape[0]
print("\n" + "=" * 98)
print("R3 — PER-TARGET TASK-LOSS MAGNITUDES (what R2 pooled)")
print("=" * 98)
print(f"  {W} windows · 32x32 · per step · eval mode · target_weights all 1.0 (C-87)")
print()
print(f"  {'target':<14}{'pos frac':>11}{'1/pos':>8}{'body nb':>11}{'body trunc':>12}"
      f"{'trunc/nb':>10}{'gate':>9}{'0.5b/g':>9}")
for j, nm in enumerate(reg_names):
    p = np.median(pf[:, j])
    bn, bt, gt = np.median(cb["nb"][:, j]), np.median(cb["trunc"][:, j]), np.median(cg[:, j])
    print(f"  {nm:<14}{p:>11.6f}{1 / p:>8.0f}{bn:>11.4f}{bt:>12.4f}"
          f"{bt / bn:>10.1f}{gt:>9.4f}{0.5 * bt / gt:>9.1f}")
print(f"  {'POOLED (R2)':<14}{np.median(pf.sum(1) / 3):>11.6f}{'':>8}"
      f"{np.median(cb['nb'].sum(1)):>11.4f}{np.median(cb['trunc'].sum(1)):>12.4f}"
      f"{np.median(cb['trunc'].sum(1)) / np.median(cb['nb'].sum(1)):>10.1f}"
      f"{np.median(cg.sum(1)):>9.4f}"
      f"{0.5 * np.median(cb['trunc'].sum(1)) / np.median(cg.sum(1)):>9.1f}")
print()
print("  HOW THE THREE BODIES SHARE THE BODY BUDGET (median share of 0.5*sum body)")
shares = (("control nb", cb["nb"]), ("control trunc", cb["trunc"]),
          ("arm trunc", ab["trunc"]))
for tag, arr in shares:
    share = np.median(arr, axis=0) / np.median(arr, axis=0).sum()
    print(f"    {tag:<16}" + "  ".join(f"{n}={s:.3f}" for n, s in zip(reg_names, share)))
print()
print("  WHAT 'A' WOULD DO — rescale each target by its OWN P_t/N (the joint hurdle NLL)")
print(f"  {'target':<14}{'as run':>11}{'under A':>11}{'gate':>9}{'A: 0.5b/g':>11}")
tot_a = 0.0
for j, nm in enumerate(reg_names):
    p = np.median(pf[:, j])
    bt, gt = np.median(ab["trunc"][:, j]), np.median(ag[:, j])
    a = bt * p
    tot_a += a
    print(f"  {nm:<14}{0.5 * bt:>11.4f}{0.5 * a:>11.4f}{gt:>9.4f}{0.5 * a / gt:>11.3f}")
print(f"  {'SUM':<14}{0.5 * np.median(ab['trunc'].sum(1)):>11.4f}{0.5 * tot_a:>11.4f}"
      f"{np.median(ag.sum(1)):>9.4f}{0.5 * tot_a / np.median(ag.sum(1)):>11.3f}")
print(f"  control reference for the same ratio: "
      f"{0.5 * np.median(cb['nb'].sum(1)) / np.median(cg.sum(1)):.3f}")
print("=" * 98)

np.savez(SP / "r3_result.npz", ctrl_nb=cb["nb"], ctrl_trunc=cb["trunc"], arm_trunc=ab["trunc"],
         ctrl_gate=cg, arm_gate=ag, pos_frac=pf, arm_pos_frac=apf,
         reg_names=np.array(reg_names), cls_names=np.array(cls_names))
print("wrote r3_result.npz")
