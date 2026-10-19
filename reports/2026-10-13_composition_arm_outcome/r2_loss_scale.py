"""R2 — the two task-loss magnitudes the frozen balancer adds together.

NOT pre-registered as an experiment: this is a descriptive measurement of numbers the engine
computes and never records. `training_engine.py` accumulates `lesson_reg` / `lesson_cls`
(:1478-1479) but writes them only to the OPT-IN trajectory CSV (:1542-1553), and
`trajectory_log_path` was unset on both runs — so neither run.log nor the offline wandb store
carries the split. It has to be recomputed.

WHAT IS COMBINED, AND HOW
-------------------------
`MultiTaskLoss` (mtloss.py:61-65) with `log_vars` frozen at 0 reduces to a constant:

    total = sum_tasks  1/((is_regression + 1) * 1) * loss_task  +  log(1)
          = 0.5 * (body_sb + body_ns + body_os)  +  1.0 * (gate_sb + gate_ns + gate_os)

Every model in the fleet pins `freeze_multitask_balancer: True`, so this is what ran.
Nothing in that expression reads the magnitudes.

WHAT IS MEASURED, on the REAL training windows
----------------------------------------------
The denominator is the point, so the population must be the one training actually saw: 32x32
windows drawn by `VolumeSampler` under `CurriculumLearner`, from the calibration train volume
(months 121-456 minus the 36-step horizon = 121-420). A number computed on the full land field
would use a denominator training never used.

Three columns, two forwards:

  (a) CONTROL weights + `nb`            — the body term the control actually paid
  (b) CONTROL weights + `truncated_nb`  — the SAME predictions, scored by the other family.
                                          Isolates the reduction change with nothing else moving.
  (c) ARM weights + `truncated_nb`      — what the arm actually paid

The gate term is identical in form on both sides (`weighted_bce`, pos_weight 2.0, full window)
and is reported per model as the reference scale the body term is added to.

SCOPE, stated rather than discovered
------------------------------------
* `model.eval()`: dropout off and BatchNorm reads its running buffers. Training ran in `train()`
  mode. Using `train()` here would rewrite the artifacts' BN buffers (C-184/C-328), so it is not
  an option; the numbers are therefore the eval-mode ones and the RATIO is what this measures.
* The sampler's RNG stream is not the production one (production draws 900 windows in sequence;
  this draws a spread across the curriculum), so the windows are not the same windows. The
  distribution they are drawn from is.
* `random_flips` is skipped — a flip relabels cells, it does not change a loss magnitude.
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

N_SAMPLED_LESSONS = 12          # x windows_per_lesson windows, spread across the curriculum
MIN_LESSONS = 300
POS_WEIGHT = 2.0                # both configs, config_hyperparameters.py:141


def load_hp(model_name: str) -> dict:
    """The config the pod trained, read from views-models rather than restated here."""
    path = MODELS / model_name / "configs" / "config_hyperparameters.py"
    spec = importlib.util.spec_from_file_location(f"hp_{model_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_hp_config()


def load_weights(pt: Path, expect_dist: str):
    """Load a delivered artifact, refusing anything that is not the 300-lesson vehicle.

    Deliberately NOT `vehicle.load_vehicle`: its F6 check pins `output_distribution: 'nb'`, which
    the arm legitimately fails. The checks that still apply are kept.
    """
    from views_hydranet.utils.utils import choose_model

    sidecar = pt.with_suffix(".pt.config.json")
    if not pt.exists() or not sidecar.exists():
        raise SystemExit(f"VOID: missing {pt} or its sidecar.")
    cfg = json.loads(sidecar.read_text())
    if cfg.get("output_distribution") != expect_dist:
        raise SystemExit(
            f"VOID: {pt.name} sidecar says output_distribution="
            f"{cfg.get('output_distribution')!r}, expected {expect_dist!r}."
        )
    sha = hashlib.sha256(pt.read_bytes()).hexdigest()
    sha_file = pt.with_suffix(".pt.sha256")
    if sha_file.exists() and sha_file.read_text().split()[0] != sha:
        raise SystemExit(f"VOID: sha256 mismatch for {pt.name}.")
    prov = {}
    pf = pt.parent / "PROVENANCE"
    if pf.exists():
        for line in pf.read_text().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                prov[k.strip()] = v.strip()
    lessons = prov.get("lessons", "?")
    if pf.exists() and (not lessons.isdigit() or int(lessons) < MIN_LESSONS):
        raise SystemExit(f"VOID: PROVENANCE lessons={lessons!r}, expected >= {MIN_LESSONS}.")
    model = choose_model(cfg, torch.device("cpu"))
    model.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    model.hurdle_nb_theta = cfg.get("hurdle_nb_theta")
    model.hurdle_lognormal_sigma = cfg.get("hurdle_lognormal_sigma")
    model.to("cpu").eval()
    print(f"  {pt.name}  sha {sha[:16]}…  dist={cfg['output_distribution']}  "
          f"lessons={lessons}  git={prov.get('git', '?')}")
    return model, cfg


def build_handler(config: dict):
    """The calibration train partition, through the production FeatureScaler and VolumeHandler."""
    from views_hydranet.utils.feature_scaler import FeatureScaler
    from views_hydranet.utils.volume_handler import VolumeHandler

    df = pull((121, 504), DIAG / "pull_land_all3_121_504.parquet")
    df = df[(df.month_id >= 121) & (df.month_id <= 456)].copy()
    r, c = grid_rowcol(df.priogrid_id.to_numpy())
    df["row"], df["col"] = r, c
    df["c_id"] = 0            # an identity channel this measurement never reads
    df = df.rename(columns=dict(zip(TARGETS, config["regression_targets"])))
    scaled = FeatureScaler(config).fit_transform(df)
    return VolumeHandler.from_df(scaled, config)


def window_losses(model, config, handler, families: dict[str, object]):
    """Per-window task losses on the windows the sampler draws, as the trainer computes them."""
    from views_hydranet.distributions.family_loss import FamilyLoss
    from views_hydranet.train.training_engine import _SequenceIndices
    from views_hydranet.utils.curriculum import CurriculumLearner
    from views_hydranet.utils.volume_sampler import VolumeSampler
    from views_hydranet.utils.weighted_bce_loss import WeightedBCEWithLogitsLoss

    sampler = VolumeSampler(handler, config)
    planner = CurriculumLearner(config, handler)
    gate_loss = WeightedBCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    body = {k: FamilyLoss(v) for k, v in families.items()}
    npar = {k: v.n_params for k, v in body.items()}
    if len(set(npar.values())) != 1:
        raise SystemExit(f"n_params differ across families {npar} — the slice is not shared.")
    npar = next(iter(npar.values()))

    wpl = config["windows_per_lesson"]
    lessons = np.unique(np.linspace(0, config["total_lessons"] - 1, N_SAMPLED_LESSONS).astype(int))
    acc = {k: [] for k in body}
    acc["gate"] = []
    acc["pos_frac"] = []
    acc["cells"] = []
    for li in lessons:
        for wi in range(wpl):
            target, threshold = planner.get_lesson(int(li) * wpl + wi)
            batch, _ = sampler.get_batch(target, threshold, batch_size=1)
            sh = batch[0]
            tt = sh.to_pytorch(torch.device("cpu"), include_identities=False)
            names = [n for n in sh.channel_map if n in sh.tensor_cols]
            idx = _SequenceIndices(names, config)
            h = model.init_hTtime(hidden_channels=model.base, H=tt.shape[-2], W=tt.shape[-1])
            per = {k: 0.0 for k in body}
            per["gate"] = 0.0
            pos = n = 0
            with torch.no_grad():
                for i in range(tt.shape[1] - 1):
                    t1 = tt[:, i + 1]
                    y_reg, y_cls = t1[:, idx.reg], t1[:, idx.cls]
                    out = model(tt[:, i][:, idx.feat], h)
                    h = out.h_next
                    for j in range(idx.n_reg):
                        p = out.reg[:, j * npar:(j + 1) * npar].permute(0, 2, 3, 1)
                        for k, fn in body.items():
                            per[k] += float(fn(p, y_reg[:, j]))
                    for j in range(idx.n_cls):
                        per["gate"] += float(gate_loss(out.cls[:, j], y_cls[:, j]))
                    pos += int((y_reg > 0).sum())
                    n += int(y_reg.numel())
            for k, v in per.items():
                acc[k].append(v / (tt.shape[1] - 1))     # per step, 3 targets summed
            acc["pos_frac"].append(pos / n)
            acc["cells"].append(tt.shape[-1] * tt.shape[-2])
    return {k: np.asarray(v) for k, v in acc.items()}, lessons, wpl


ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
print("loading vehicles …")
ctrl, ctrl_sidecar = load_weights(CONTROL_PT, "nb")
arm, arm_sidecar = load_weights(ARM_PT, "truncated_nb")

ctrl_cfg, arm_cfg = load_hp("violet_visitor"), load_hp("amber_kestrel")
for key in ("freeze_multitask_balancer", "loss_class_pos_weight", "window_dim",
            "windows_per_lesson", "total_lessons", "sampling_strategy", "min_events"):
    if ctrl_cfg.get(key) != arm_cfg.get(key):
        raise SystemExit(f"VOID: configs differ on {key}: "
                         f"{ctrl_cfg.get(key)!r} vs {arm_cfg.get(key)!r}")
if not ctrl_cfg["freeze_multitask_balancer"]:
    raise SystemExit("VOID: the control did NOT freeze the balancer — the constant form is wrong.")

from views_hydranet.distributions import resolve_family  # noqa: E402

print("building the calibration train volume (121-456, minus the 36-step horizon) …")
handler = build_handler(ctrl_cfg)

print(f"\nsampling {N_SAMPLED_LESSONS} lessons x {ctrl_cfg['windows_per_lesson']} windows "
      f"at {ctrl_cfg['window_dim']}x{ctrl_cfg['window_dim']} …")
print("  control weights, both families on the same predictions …")
c, lessons, wpl = window_losses(
    ctrl, ctrl_cfg, handler,
    {"nb": resolve_family("nb"), "trunc": resolve_family("truncated_nb")},
)
print("  arm weights, its own family …")
a, _, _ = window_losses(arm, arm_cfg, handler, {"trunc": resolve_family("truncated_nb")})

W = len(c["gate"])
print("\n" + "=" * 90)
print("R2 — TASK-LOSS MAGNITUDES ON THE REAL TRAINING WINDOWS")
print("=" * 90)
print(f"  {W} windows · {ctrl_cfg['window_dim']}x{ctrl_cfg['window_dim']} = "
      f"{c['cells'][0]} cells · per-step, 3 targets summed · eval mode")
print(f"  positive cell fraction in the sampled windows: median {np.median(c['pos_frac']):.6f}  "
      f"(mean {c['pos_frac'].mean():.6f}, range "
      f"{c['pos_frac'].min():.6f}-{c['pos_frac'].max():.6f})")
print(f"  => the truncated reduction's denominator is 1/{1 / np.median(c['pos_frac']):.0f} "
      f"of the dense one")
print()
rows = [
    ("CONTROL nb        (a) body", c["nb"], 0.5),
    ("CONTROL truncated (b) body", c["trunc"], 0.5),
    ("ARM     truncated (c) body", a["trunc"], 0.5),
    ("CONTROL           gate", c["gate"], 1.0),
    ("ARM               gate", a["gate"], 1.0),
]
print(f"  {'term':<28}{'median':>12}{'mean':>12}{'coeff':>8}{'median x coeff':>16}")
for name, v, co in rows:
    print(f"  {name:<28}{np.median(v):>12.4f}{v.mean():>12.4f}{co:>8.1f}"
          f"{np.median(v) * co:>16.4f}")
print()
print("  THE BALANCE EACH RUN ACTUALLY ADDED UP (median per step):")
for tag, b, g in (("control", c["nb"], c["gate"]), ("arm", a["trunc"], a["gate"])):
    bb, gg = 0.5 * np.median(b), 1.0 * np.median(g)
    print(f"    {tag:<8} 0.5*body {bb:>10.4f}  +  1.0*gate {gg:>8.4f}   "
          f"=> body is {bb / gg:>8.1f}x the gate term")
print()
print("  THE ONE VARIABLE, isolated on identical predictions (control weights):")
bn, bt = np.median(c["nb"]), np.median(c["trunc"])
print(f"    nb {bn:.4f} -> truncated_nb {bt:.4f}   ratio {bt / bn:.1f}x")
print("=" * 90)

np.savez(
    SP / "r2_result.npz",
    ctrl_nb=c["nb"], ctrl_trunc=c["trunc"], ctrl_gate=c["gate"],
    arm_trunc=a["trunc"], arm_gate=a["gate"],
    pos_frac=c["pos_frac"], cells=c["cells"], lessons=lessons,
    windows_per_lesson=wpl,
)
print("wrote r2_result.npz")
